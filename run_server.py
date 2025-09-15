#!/usr/bin/env python
"""
Standalone script to run the Odoo MCP server 
Optimized for Docker container environment
"""
import sys
import os
import asyncio
import anyio
import logging
import datetime
from pathlib import Path

from mcp.server.stdio import stdio_server
from mcp.server.lowlevel import Server
import mcp.types as types

from odoo_mcp.server import mcp  # FastMCP instance from our code

# Safe import for langgraph with fallback
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def setup_logging():
    """Set up logging optimized for Docker container"""
    # In Docker, we want to log to stdout/stderr for Docker logging
    # But also keep file logging if the volume is mounted

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO)

    # Console handler (this goes to Docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler only if logs directory exists (volume mounted)
    log_dir = Path("/app/logs")
    if log_dir.exists() and log_dir.is_dir():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"mcp_server_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Format for file handler (more detailed)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler.setLevel(logging.INFO)  # Less verbose for console in Docker

    # Format for console handler (simpler for Docker logs)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_db_config():
    """Get database configuration from environment variables"""
    config = {}

    # Required variables
    required_vars = {
        "DB_USER": "odoo",  # default value
        "DB_PASSWORD": None,  # no default, must be set
        "DB_HOST": "db",  # default to docker service name
        "DB_NAME": "odoo",  # default value
        "DB_PORT": "5432",  # default PostgreSQL port
    }

    missing_vars = []

    for var_name, default in required_vars.items():
        value = os.getenv(var_name, default)
        if value is None:
            missing_vars.append(var_name)
        else:
            config[var_name.lower()] = value

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Optional variables
    config["pgpassword"] = os.getenv("PGPASSWORD", config["db_password"])

    # Construct PG_URI if not provided
    pg_uri = os.getenv("PG_URI")
    if not pg_uri:
        pg_uri = f"postgresql://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"

    config["pg_uri"] = pg_uri

    return config


def main(transport="streamable-http") -> int:
    """
    Run the MCP server in Docker container environment
    """
    logger = setup_logging()

    try:
        logger.info("=== ODOO MCP SERVER STARTING IN DOCKER ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Container working directory: {Path.cwd()}")

        # Get database configuration
        try:
            db_config = get_db_config()
            logger.info("Database configuration loaded:")
            logger.info(f"  DB_HOST: {db_config['db_host']}")
            logger.info(f"  DB_PORT: {db_config['db_port']}")
            logger.info(f"  DB_USER: {db_config['db_user']}")
            logger.info(f"  DB_NAME: {db_config['db_name']}")
            logger.info(
                f"  DB_PASSWORD: {'***set***' if db_config['db_password'] else 'NOT SET'}"
            )

        except ValueError as e:
            logger.error(f"Database configuration error: {e}")
            logger.info("Available environment variables:")
            for key in sorted(os.environ.keys()):
                if any(prefix in key for prefix in ["DB_", "ODOO_", "PG_", "MCP_"]):
                    if "PASSWORD" in key:
                        logger.info(f"  {key}=***hidden***")
                    else:
                        logger.info(f"  {key}={os.environ[key]}")
            return 1

        # Log other important environment variables
        logger.info("Odoo configuration:")
        logger.info(f"  ODOO_URL: {os.getenv('ODOO_URL', 'NOT SET')}")
        logger.info(f"  ODOO_DB: {os.getenv('ODOO_DB', 'NOT SET')}")
        logger.info(f"  ODOO_USERNAME: {os.getenv('ODOO_USERNAME', 'NOT SET')}")
        logger.info(
            f"  ODOO_PASSWORD: {'***set***' if os.getenv('ODOO_PASSWORD') else 'NOT SET'}"
        )

        logger.info("MCP configuration:")
        logger.info(f"  MCP_TRANSPORT: {os.getenv('MCP_TRANSPORT', 'streamable-http')}")
        logger.info(f"  DEBUG: {os.getenv('DEBUG', '0')}")
        logger.info(
            f"  OPENAI_API_KEY: {'***set***' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}"
        )

        logger.info(f"MCP object type: {type(mcp)}")

        async def asetup_checkpointer():
            """Initialize PostgreSQL checkpoint saver"""
            try:
                logger.info("Initializing PostgreSQL checkpoint saver...")
                # Use the constructed PG_URI
                masked_uri = db_config["pg_uri"].replace(
                    db_config["db_password"], "***"
                )
                logger.info(f"Using PostgreSQL URI: {masked_uri}")

                async with AsyncPostgresSaver.from_conn_string(
                    db_config["pg_uri"]
                ) as saver:
                    await saver.setup()
                    logger.info("PostgreSQL checkpoint saver initialized successfully")

                logger.info("PostgreSQL checkpoint saver cleaned up")
                return
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL saver: {str(e)}")
                logger.error("This may be normal if the database is not ready yet")

        # Determine transport from environment
        transport_mode = os.getenv("MCP_TRANSPORT", transport)

        if transport_mode == "streamable-http":
            logger.info("Using streamable-http transport")
            asyncio.run(asetup_checkpointer())
            mcp.run(transport="streamable-http")
            logger.info("MCP server stopped normally")
            return 0
        else:
            logger.info("Using stdio transport")

            # Run server in stdio mode
            async def arun():
                logger.info("Starting Odoo MCP server with stdio transport...")
                async with stdio_server() as streams:
                    logger.info("Stdio server initialized, running MCP server...")
                    await mcp._mcp_server.run(
                        streams[0],
                        streams[1],
                        mcp._mcp_server.create_initialization_options(),
                    )

            # Run server
            anyio.run(arun)
            logger.info("MCP server stopped normally")
            return 0

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # In Docker, we want to handle signals properly
    import signal

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    sys.exit(main())
