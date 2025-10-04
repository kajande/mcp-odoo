"""
MCP server for Odoo integration
Provides MCP tools and resources for interacting with Odoo ERP systems
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

from mcp.server.fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

from .odoo_client import OdooClient, get_odoo_client

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from .prompts import EXECUTE_METHOD_DESCRIPTION


@dataclass
class AppContext:
    """Application context for the MCP server"""

    odoo: OdooClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Application lifespan for initialization and cleanup
    """
    # Initialize Odoo client on startup
    odoo_client = get_odoo_client()

    try:
        yield AppContext(odoo=odoo_client)
    finally:
        # No cleanup needed for Odoo client
        pass


# Determine transport configuration
transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
stateless_http = transport == "streamable-http"
host = "0.0.0.0" if stateless_http else "127.0.0.1"
port = 8000 if stateless_http else None

logger.info(f"Creating FastMCP with transport: {transport}")
logger.info(f"HTTP server will bind to: {host}:{port}")

# Create MCP server with transport configuration
mcp = FastMCP(
    "Odoo MCP Server",
    dependencies=["requests"],
    lifespan=app_lifespan,
    host=host,
    port=port,
    stateless_http=stateless_http,
)


# ----- MCP Resources -----


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    return JSONResponse({"status": "healthy"})


@mcp.resource(
    "odoo://models", description="List all available models in the Odoo system"
)
def get_models() -> str:
    """Lists all available models in the Odoo system"""
    odoo_client = get_odoo_client()
    models = odoo_client.get_models()
    return json.dumps(models, indent=2)


@mcp.resource(
    "odoo://model/{model_name}",
    description="Get detailed information about a specific model including fields",
)
def get_model_info(model_name: str) -> str:
    """
    Get information about a specific model

    Parameters:
        model_name: Name of the Odoo model (e.g., 'res.partner')
    """
    odoo_client = get_odoo_client()
    try:
        # Get model info
        model_info = odoo_client.get_model_info(model_name)

        # Get field definitions
        fields = odoo_client.get_model_fields(model_name)
        model_info["fields"] = fields

        return json.dumps(model_info, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://record/{model_name}/{record_id}",
    description="Get detailed information of a specific record by ID",
)
def get_record(model_name: str, record_id: str) -> str:
    """
    Get a specific record by ID

    Parameters:
        model_name: Name of the Odoo model (e.g., 'res.partner')
        record_id: ID of the record
    """
    odoo_client = get_odoo_client()
    try:
        record_id_int = int(record_id)
        record = odoo_client.read_records(model_name, [record_id_int])
        if not record:
            return json.dumps(
                {"error": f"Record not found: {model_name} ID {record_id}"}, indent=2
            )
        return json.dumps(record[0], indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://search/{model_name}/{domain}",
    description="Search for records matching the domain",
)
def search_records_resource(model_name: str, domain: str) -> str:
    """
    Search for records that match a domain

    Parameters:
        model_name: Name of the Odoo model (e.g., 'res.partner')
        domain: Search domain in JSON format (e.g., '[["name", "ilike", "test"]]')
    """
    odoo_client = get_odoo_client()
    try:
        # Parse domain from JSON string
        domain_list = json.loads(domain)

        # Set a reasonable default limit
        limit = 10

        # Perform search_read for efficiency
        results = odoo_client.search_read(model_name, domain_list, limit=limit)

        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ----- Pydantic models for type safety -----


class DomainCondition(BaseModel):
    """A single condition in a search domain"""

    field: str = Field(description="Field name to search")
    operator: str = Field(
        description="Operator (e.g., '=', '!=', '>', '<', 'in', 'not in', 'like', 'ilike')"
    )
    value: Any = Field(description="Value to compare against")

    def to_tuple(self) -> List:
        """Convert to Odoo domain condition tuple"""
        return [self.field, self.operator, self.value]


class SearchDomain(BaseModel):
    """Search domain for Odoo models"""

    conditions: List[DomainCondition] = Field(
        default_factory=list,
        description="List of conditions for searching. All conditions are combined with AND operator.",
    )

    def to_domain_list(self) -> List[List]:
        """Convert to Odoo domain list format"""
        return [condition.to_tuple() for condition in self.conditions]


# ----- MCP Tools -----


@mcp.tool(description=EXECUTE_METHOD_DESCRIPTION)
def execute_method(
    ctx: Context,
    model: str,
    method: str,
    args: List = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a custom method on an Odoo model

    Parameters:
        model: The model name (e.g., 'res.partner')
        method: Method name to execute
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - result: Result of the method (if success)
        - error: Error message (if failure)
    """
    odoo = ctx.request_context.lifespan_context.odoo
    try:
        args = args or []
        kwargs = kwargs or {}

        # Special handling for search methods like search, search_count, search_read
        search_methods = ["search", "search_count", "search_read"]
        if method in search_methods and args:
            # Search methods usually have domain as the first parameter
            # args: [[domain], limit, offset, ...] or [domain, limit, offset, ...]
            normalized_args = list(
                args
            )  # Create a copy to avoid affecting the original args

            if len(normalized_args) > 0:
                # Process domain in args[0]
                domain = normalized_args[0]
                domain_list = []

                # Check if domain is wrapped unnecessarily ([domain] instead of domain)
                if (
                    isinstance(domain, list)
                    and len(domain) == 1
                    and isinstance(domain[0], list)
                ):
                    # Case [[domain]] - unwrap to [domain]
                    domain = domain[0]

                # Normalize domain similar to search_records function
                if domain is None:
                    domain_list = []
                elif isinstance(domain, dict):
                    if "conditions" in domain:
                        # Object format
                        conditions = domain.get("conditions", [])
                        domain_list = []
                        for cond in conditions:
                            if isinstance(cond, dict) and all(
                                k in cond for k in ["field", "operator", "value"]
                            ):
                                domain_list.append(
                                    [cond["field"], cond["operator"], cond["value"]]
                                )
                elif isinstance(domain, list):
                    # List format
                    if not domain:
                        domain_list = []
                    elif all(isinstance(item, list) for item in domain) or any(
                        item in ["&", "|", "!"] for item in domain
                    ):
                        domain_list = domain
                    elif len(domain) >= 3 and isinstance(domain[0], str):
                        # Case [field, operator, value] (not [[field, operator, value]])
                        domain_list = [domain]
                elif isinstance(domain, str):
                    # String format (JSON)
                    try:
                        parsed_domain = json.loads(domain)
                        if (
                            isinstance(parsed_domain, dict)
                            and "conditions" in parsed_domain
                        ):
                            conditions = parsed_domain.get("conditions", [])
                            domain_list = []
                            for cond in conditions:
                                if isinstance(cond, dict) and all(
                                    k in cond for k in ["field", "operator", "value"]
                                ):
                                    domain_list.append(
                                        [cond["field"], cond["operator"], cond["value"]]
                                    )
                        elif isinstance(parsed_domain, list):
                            domain_list = parsed_domain
                    except json.JSONDecodeError:
                        try:
                            import ast

                            parsed_domain = ast.literal_eval(domain)
                            if isinstance(parsed_domain, list):
                                domain_list = parsed_domain
                        except:
                            domain_list = []

                # Xác thực domain_list
                if domain_list:
                    valid_conditions = []
                    for cond in domain_list:
                        if isinstance(cond, str) and cond in ["&", "|", "!"]:
                            valid_conditions.append(cond)
                            continue

                        if (
                            isinstance(cond, list)
                            and len(cond) == 3
                            and isinstance(cond[0], str)
                            and isinstance(cond[1], str)
                        ):
                            valid_conditions.append(cond)

                    domain_list = valid_conditions

                # Cập nhật args với domain đã chuẩn hóa
                normalized_args[0] = domain_list
                args = normalized_args

                # Log for debugging
                print(f"Executing {method} with normalized domain: {domain_list}")

        result = odoo.execute_method(model, method, *args, **kwargs)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}



if __name__ == '__main__':
    # Example usage of the execute_method tool with fast_service module
    print("Odoo MCP Server - Example Usage for Fast Service Module")
    print("=" * 60)
    
    # # Example 1: Search for services
    # example1 = {
    #     "model": "fast_service.service",
    #     "method": "search_read",
    #     "args": [
    #         [["name", "ilike", "Dulayni"]],  # Domain to find Dulayni service
    #         ["name", "service_type", "minimum_topup", "queue"]  # Fields to return
    #     ]
    # }
    
    # print("\n1. Search for Dulayni Service:")
    # result = execute_method(None, **example1)
    # print(result)
    
    # # Example 2: Get product catalog for a service
    # example2 = {
    #     "model": "fast_service.service",
    #     "method": "read",
    #     "args": [[1]],  # Assuming service ID 1 is Dulayni
    #     "kwargs": {
    #         "fields": ["name", "product_ids"]
    #     }
    # }
    
    # print("\n2. Get Service Product Catalog:")
    # result = execute_method(None, **example2)
    # print(result)
    
    # # Example 3: Create a new order
    # example3 = {
    #     "model": "fast_service.order",
    #     "method": "create",
    #     "args": [{
    #         "customer_id": 1,  # Partner ID
    #         "service_id": 1,   # Service ID
    #         "state": "draft",
    #         "order_line_ids": [
    #             (0, 0, {
    #                 "product_id": 1,  # gpt-4o product
    #                 "quantity": 2
    #             }),
    #             (0, 0, {
    #                 "product_id": 2,  # gpt-4o-mini product
    #                 "quantity": 1
    #             })
    #         ]
    #     }]
    # }
    
    # result = execute_method(None, **example3)
    # print(result)


    # # Example 4: Search orders with timing statistics
    # example4 = {
    #     "model": "fast_service.order",
    #     "method": "search_read",
    #     "args": [
    #         [["state", "=", "delivered"]],  # Completed orders
    #         ["name", "customer_id", "total", "completion_time", "start_time", "end_time"],
    #         10,  # Limit
    #         0    # Offset
    #     ]
    # }
    
    # print("\n4. Get Completed Orders with Timing:")
    # result = execute_method(None, **example4)
    # print(result)

    # # Example 5: Update service performance metrics
    # example5 = {
    #     "model": "fast_service.service",
    #     "method": "write",
    #     "args": [
    #         [1],  # Service ID
    #         {
    #             "minimum_topup": 1000.0,  # Update minimum top-up
    #         }
    #     ]
    # }
    
    # print("\n5. Update Service Settings:")
    # result = execute_method(None, **example5)
    # print(result)

    
    # # Example 6: Send whatsapp message (via execute_method)
    # example6 = {
    #     "model": "whatsapp.service",
    #     "method": "send_message",
    #     "args": [
    #         [1],
    #         "+221778577500",  # Phone number
    #         "1. gpt-4o - 15.0 XOF\n2. gpt-4o-mini - 10.0 XOF",   # Message
    #     ]
    # }
    # result = execute_method(None, **example6)
    # print(result)

    # # Example 7: Rquest for payment (via execute_method)
    # example7 = {
    #     "model": "wave.service",
    #     "method": "create_checkout_session",
    #     "args": [
    #         [1],
    #         "+221778577500",  # Phone number
    #         100,   # Amount
    #     ]
    # }
    # result = execute_method(None, **example7)
    # print(result['result'].get('wave_launch_url'))

    # example76 = {
    #     "model": "whatsapp.service",
    #     "method": "send_message",
    #     "args": [
    #         [1],
    #         "+221778577500",  # Phone number
    #         f"{result['result'].get('wave_launch_url')}",   # Message
    #     ]
    # }
    # result = execute_method(None, **example76)
    # print(result)
    
    
    # # Example 9: Get service performance analytics
    # example9 = {
    #     "model": "fast_service.service",
    #     "method": "search_read",
    #     "args": [
    #         [["service_type", "=", "other"]],  # AI services
    #         ["name", "queue", "min_completion_time", "max_completion_time", "avg_completion_time", "total_completed_orders"]
    #     ]
    # }
    
    # print("\n9. Get Service Performance Analytics:")
    # print(f"   execute_method({json.dumps(example9, indent=2)})")
