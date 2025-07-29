#!/usr/bin/env python
"""
Standalone script to run the Odoo MCP server 
Uses the same approach as in the official MCP SDK examples
"""
import sys
import os
import asyncio
import anyio
import logging
import datetime

from mcp.server.stdio import stdio_server
from mcp.server.lowlevel import Server
import mcp.types as types

from odoo_mcp.server import mcp  # FastMCP instance from our code

# Safe import for langgraph with fallback
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

PG_URI_PROD = 'postgresql://postgres.japcfankaqxrwyjzydsy:"ciyrF86sP9gH&-J"@aws-0-eu-central-1.pooler.supabase.com:5432/postgres'
PG_URI_DEV = "postgresql://odoo:odoo@db:5432/odoo"

def setup_logging():
    """Set up logging to both console and file"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mcp_server_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def main(transport='streamable-http') -> int:
    """
    Run the MCP server based on the official examples
    """
    logger = setup_logging()
    
    try:
        logger.info("=== ODOO MCP SERVER STARTING ===")
        logger.info(f"Python version: {sys.version}")
        logger.info("Environment variables:")
        for key, value in os.environ.items():
            if key.startswith("ODOO_"):
                if key == "ODOO_PASSWORD":
                    logger.info(f"  {key}: ***hidden***")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info(f"MCP object type: {type(mcp)}")

        async def asetup_checkpointer():
            # Initialize PostgreSQL checkpoint saver if available
            saver = None
            try:
                # Get PostgreSQL connection string from environment
                PG_URI = os.getenv("PG_URI", PG_URI_DEV)
                
                logger.info("Initializing PostgreSQL checkpoint saver...")
                logger.info(f"Using PostgreSQL URI: {PG_URI}")
                
                # Use context manager as in the working example
                async with AsyncPostgresSaver.from_conn_string(PG_URI) as saver:
                    await saver.setup()
                    logger.info("PostgreSQL checkpoint saver initialized successfully")

                logger.info("PostgreSQL checkpoint saver cleaned up")
                return
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL saver: {str(e)}")

        
        if transport == 'streamable-http':
            asyncio.run(asetup_checkpointer())
            mcp.run(transport='streamable-http')
            logger.info("MCP server stopped normally")
            return 0
        # Run server in stdio mode like the official examples
        async def arun():
            logger.info("Starting Odoo MCP server with stdio transport...")
            async with stdio_server() as streams:
                logger.info("Stdio server initialized, running MCP server...")
                await mcp._mcp_server.run(
                    streams[0], streams[1], mcp._mcp_server.create_initialization_options()
                )
                
        # Run server
        anyio.run(arun)
        logger.info("MCP server stopped normally")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
