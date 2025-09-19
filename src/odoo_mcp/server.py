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


class EmployeeSearchResult(BaseModel):
    """Represents a single employee search result."""

    id: int = Field(description="Employee ID")
    name: str = Field(description="Employee name")


class SearchEmployeeResponse(BaseModel):
    """Response model for the search_employee tool."""

    success: bool = Field(description="Indicates if the search was successful")
    result: Optional[List[EmployeeSearchResult]] = Field(
        default=None, description="List of employee search results"
    )
    error: Optional[str] = Field(default=None, description="Error message, if any")


class Holiday(BaseModel):
    """Represents a single holiday."""

    display_name: str = Field(description="Display name of the holiday")
    start_datetime: str = Field(description="Start date and time of the holiday")
    stop_datetime: str = Field(description="End date and time of the holiday")
    employee_id: List[Union[int, str]] = Field(
        description="Employee ID associated with the holiday"
    )
    name: str = Field(description="Name of the holiday")
    state: str = Field(description="State of the holiday")


class SearchHolidaysResponse(BaseModel):
    """Response model for the search_holidays tool."""

    success: bool = Field(description="Indicates if the search was successful")
    result: Optional[List[Holiday]] = Field(
        default=None, description="List of holidays found"
    )
    error: Optional[str] = Field(default=None, description="Error message, if any")


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

# Add these imports at the top of server.py if not already present
from typing import List, Dict, Any, Optional

# Add these description constants after the existing EXECUTE_METHOD_DESCRIPTION
GET_MENU_DESCRIPTION = """
Get the menu/catalog of products for a specific fast service. This tool retrieves all available products 
for a given service, including their names, descriptions, and prices.

## When to Use This Tool

Use this tool when:
1. **Customer Inquiries**: When a customer wants to see what products/services are available
2. **Menu Display**: When you need to show the complete catalog for a service
3. **Product Information**: When you need to get product details for order processing
4. **Service Validation**: When you need to verify if a service exists and has products

## Parameters

- **service_name**: The name of the service (e.g., "Dulayni", "Restaurant ABC")

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **service_info**: Service details (name, type, minimum_topup)
- **products**: List of products with id, name, description, and price
- **error**: Error message if the operation failed

## Example Usage

```python
result = get_menu("Dulayni")
# Returns: {
#   "success": True,
#   "service_info": {"name": "Dulayni", "type": "other", "minimum_topup": 500.0},
#   "products": [
#     {"id": 1, "name": "gpt-4o", "description": "OpenAI's most advanced...", "price": 15.0},
#     {"id": 2, "name": "claude-sonnet-4", "description": "Balanced Claude model...", "price": 12.0}
#   ]
# }
```
"""

MAKE_ORDER_DESCRIPTION = """
Create a new order for a fast service. This tool handles the complete order creation process including 
customer lookup/creation, order line creation, and initial order setup.

## When to Use This Tool

Use this tool when:
1. **Order Placement**: When a customer wants to place a new order
2. **Bulk Ordering**: When processing multiple items in a single order
3. **Customer Order Management**: When you need to create orders for existing or new customers
4. **E-commerce Integration**: When integrating with external ordering systems

## Parameters

- **service_name**: The name of the service
- **customer_phone**: Customer's phone number (used for identification)
- **customer_name**: Customer's name (optional, will be auto-generated if not provided)
- **order_items**: List of dictionaries with product_id and quantity

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **order_id**: ID of the created order
- **order_reference**: Human-readable order reference (e.g., "ORD00001")
- **total_amount**: Total amount of the order
- **queue_position**: Current position in the service queue
- **error**: Error message if the operation failed

## Example Usage

```python
result = make_order(
    service_name="Dulayni",
    customer_phone="+221781234567",
    customer_name="John Doe",
    order_items=[
        {"product_id": 1, "quantity": 2},
        {"product_id": 3, "quantity": 1}
    ]
)
# Returns: {
#   "success": True,
#   "order_id": 123,
#   "order_reference": "ORD00001",
#   "total_amount": 45.0,
#   "queue_position": 3
# }
```
"""

GET_ORDER_STATUS_DESCRIPTION = """
Retrieve the current status and details of a specific order. This tool provides comprehensive 
information about an order's current state, progress, and details.

## When to Use This Tool

Use this tool when:
1. **Order Tracking**: When customers want to check their order status
2. **Staff Monitoring**: When staff need to see order progress
3. **Customer Service**: When handling customer inquiries about orders
4. **System Integration**: When external systems need order status updates

## Parameters

- **service_name**: The name of the service
- **order_reference**: The order reference number (e.g., "ORD00001") OR
- **order_id**: The internal order ID (alternative to order_reference)

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **order_info**: Complete order information including:
  - id, reference, customer details, service info
  - current state, total amount, payment status
  - order items with quantities and prices
  - timestamps and staff assignments
- **error**: Error message if the operation failed

## Example Usage

```python
result = get_order_status("Dulayni", order_reference="ORD00001")
# Returns: {
#   "success": True,
#   "order_info": {
#     "id": 123,
#     "reference": "ORD00001",
#     "customer": {"name": "John Doe", "phone": "+221781234567"},
#     "state": "preparing",
#     "total": 45.0,
#     "payment_status": "paid",
#     "items": [...]
#   }
# }
```
"""

GET_ORDER_POSITION_DESCRIPTION = """
Get the current queue position of an order within a service. This tool helps track where 
an order stands in the preparation queue and provides estimated timing information.

## When to Use This Tool

Use this tool when:
1. **Queue Updates**: When customers want to know their position in line
2. **Wait Time Estimation**: When providing estimated preparation times
3. **Queue Management**: When staff need to see queue status
4. **Customer Communication**: When sending automated position updates

## Parameters

- **service_name**: The name of the service
- **order_reference**: The order reference number (e.g., "ORD00001") OR
- **order_id**: The internal order ID (alternative to order_reference)

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **position_info**: Position details including:
  - current_position: Current position in queue (0 if not in queue)
  - total_in_queue: Total number of orders in queue
  - orders_ahead: Number of orders ahead of this one
  - estimated_wait: Rough estimate of wait time
- **order_state**: Current state of the order
- **error**: Error message if the operation failed

## Example Usage

```python
result = get_order_position("Dulayni", order_reference="ORD00001")
# Returns: {
#   "success": True,
#   "position_info": {
#     "current_position": 3,
#     "total_in_queue": 5,
#     "orders_ahead": 2,
#     "estimated_wait": "15-20 minutes"
#   },
#   "order_state": "confirmed"
# }
```
"""

# Add these MCP tools after the existing tools in server.py

@mcp.tool(description=GET_MENU_DESCRIPTION)
def get_menu(
    ctx: Context,
    service_name: str,
) -> Dict[str, Any]:
    """
    Get the menu/catalog of products for a specific fast service
    
    Parameters:
        service_name: The name of the service
        
    Returns:
        Dictionary containing service info, products list, or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # First, find the service by name
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {"fields": ["id", "name", "service_type", "minimum_topup"], "limit": 1}
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service = service_result[0]
        service_id = service["id"]
        
        # Get products associated with this service
        products_result = odoo.execute_method(
            "fast_service.service",
            "read",
            [service_id],
            {"fields": ["product_ids"]}
        )
        
        if not products_result or not products_result[0].get("product_ids"):
            return {
                "success": True,
                "service_info": {
                    "name": service["name"],
                    "type": service["service_type"],
                    "minimum_topup": service["minimum_topup"]
                },
                "products": []
            }
        
        product_ids = products_result[0]["product_ids"]
        
        # Get product details
        products = odoo.execute_method(
            "product.product",
            "read",
            product_ids,
            {"fields": ["id", "name", "description", "list_price"]}
        )
        
        return {
            "success": True,
            "service_info": {
                "name": service["name"],
                "type": service["service_type"],
                "minimum_topup": service["minimum_topup"]
            },
            "products": [
                {
                    "id": product["id"],
                    "name": product["name"],
                    "description": product.get("description") or "",
                    "price": product["list_price"]
                }
                for product in products
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving menu: {str(e)}"
        }


@mcp.tool(description=MAKE_ORDER_DESCRIPTION)
def make_order(
    ctx: Context,
    service_name: str,
    customer_phone: str,
    order_items: List[Dict[str, Any]],
    customer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new order for a fast service
    
    Parameters:
        service_name: The name of the service
        customer_phone: Customer's phone number
        order_items: List of dicts with product_id and quantity
        customer_name: Customer's name (optional)
        
    Returns:
        Dictionary containing order details or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Find the service
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {"fields": ["id", "name"], "limit": 1}
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service_id = service_result[0]["id"]
        
        # Find or create customer
        customer_result = odoo.execute_method(
            "res.partner",
            "search_read",
            [("mobile", "=", customer_phone)],
            {"fields": ["id", "name"], "limit": 1}
        )
        
        if customer_result:
            customer_id = customer_result[0]["id"]
        else:
            # Create new customer
            customer_name = customer_name or f"Customer {customer_phone}"
            customer_id = odoo.execute_method(
                "res.partner",
                "create",
                {
                    "name": customer_name,
                    "mobile": customer_phone
                }
            )
        
        # Create the order
        order_id = odoo.execute_method(
            "fast_service.order",
            "create",
            {
                "service_id": service_id,
                "customer_id": customer_id,
                "state": "draft"
            }
        )
        
        # Create order lines
        for item in order_items:
            odoo.execute_method(
                "fast_service.order.line",
                "create",
                {
                    "order_id": order_id,
                    "product_id": item["product_id"],
                    "quantity": item["quantity"]
                }
            )
        
        # Get the created order details
        order_result = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
            {"fields": ["name", "total", "position"]}
        )
        
        if not order_result:
            return {
                "success": False,
                "error": "Failed to retrieve created order details"
            }
        
        order = order_result[0]
        
        return {
            "success": True,
            "order_id": order_id,
            "order_reference": order["name"],
            "total_amount": order["total"],
            "queue_position": order["position"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating order: {str(e)}"
        }


@mcp.tool(description=GET_ORDER_STATUS_DESCRIPTION)
def get_order_status(
    ctx: Context,
    service_name: str,
    order_reference: Optional[str] = None,
    order_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get the status and details of a specific order
    
    Parameters:
        service_name: The name of the service
        order_reference: The order reference number (alternative to order_id)
        order_id: The internal order ID (alternative to order_reference)
        
    Returns:
        Dictionary containing order information or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Find the service first for validation
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {"fields": ["id", "name"], "limit": 1}
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service_id = service_result[0]["id"]
        
        # Build search domain based on provided parameters
        if order_id:
            domain = [("id", "=", order_id), ("service_id", "=", service_id)]
        elif order_reference:
            domain = [("name", "=", order_reference), ("service_id", "=", service_id)]
        else:
            return {
                "success": False,
                "error": "Either order_reference or order_id must be provided"
            }
        
        # Get order details
        order_result = odoo.execute_method(
            "fast_service.order",
            "search_read",
            domain,
            {
                "fields": [
                    "id", "name", "customer_id", "service_id", "date", "state",
                    "total", "position", "payment_url", "payment_status",
                    "preparer_id", "deliverer_id"
                ],
                "limit": 1
            }
        )
        
        if not order_result:
            return {
                "success": False,
                "error": "Order not found"
            }
        
        order = order_result[0]
        
        # Get order lines
        order_lines = odoo.execute_method(
            "fast_service.order.line",
            "search_read",
            [("order_id", "=", order["id"])],
            {"fields": ["product_id", "quantity", "price", "subtotal"]}
        )
        
        return {
            "success": True,
            "order_info": {
                "id": order["id"],
                "reference": order["name"],
                "customer": {
                    "id": order["customer_id"][0] if order["customer_id"] else None,
                    "name": order["customer_id"][1] if order["customer_id"] else "Unknown"
                },
                "service": {
                    "id": order["service_id"][0],
                    "name": order["service_id"][1]
                },
                "date": order["date"],
                "state": order["state"],
                "total": order["total"],
                "position": order["position"],
                "payment_status": order["payment_status"],
                "payment_url": order.get("payment_url"),
                "preparer": {
                    "id": order["preparer_id"][0] if order["preparer_id"] else None,
                    "name": order["preparer_id"][1] if order["preparer_id"] else None
                },
                "deliverer": {
                    "id": order["deliverer_id"][0] if order["deliverer_id"] else None,
                    "name": order["deliverer_id"][1] if order["deliverer_id"] else None
                },
                "items": [
                    {
                        "product_id": line["product_id"][0],
                        "product_name": line["product_id"][1],
                        "quantity": line["quantity"],
                        "price": line["price"],
                        "subtotal": line["subtotal"]
                    }
                    for line in order_lines
                ]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving order status: {str(e)}"
        }


@mcp.tool(description=GET_ORDER_POSITION_DESCRIPTION)
def get_order_position(
    ctx: Context,
    service_name: str,
    order_reference: Optional[str] = None,
    order_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get the current queue position of an order
    
    Parameters:
        service_name: The name of the service
        order_reference: The order reference number (alternative to order_id)
        order_id: The internal order ID (alternative to order_reference)
        
    Returns:
        Dictionary containing position information or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Find the service
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {"fields": ["id", "name"], "limit": 1}
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service_id = service_result[0]["id"]
        
        # Build search domain
        if order_id:
            domain = [("id", "=", order_id), ("service_id", "=", service_id)]
        elif order_reference:
            domain = [("name", "=", order_reference), ("service_id", "=", service_id)]
        else:
            return {
                "success": False,
                "error": "Either order_reference or order_id must be provided"
            }
        
        # Get the specific order
        order_result = odoo.execute_method(
            "fast_service.order",
            "search_read",
            domain,
            {"fields": ["id", "name", "state", "position", "date"], "limit": 1}
        )
        
        if not order_result:
            return {
                "success": False,
                "error": "Order not found"
            }
        
        order = order_result[0]
        
        # Get total number of orders in queue for this service
        total_in_queue = odoo.execute_method(
            "fast_service.order",
            "search_count",
            [
                ("service_id", "=", service_id),
                ("state", "in", ["confirmed", "preparing"])
            ]
        )
        
        # Calculate estimated wait time (rough estimate: 10 minutes per order ahead)
        orders_ahead = max(0, order["position"] - 1) if order["position"] > 0 else 0
        estimated_minutes = orders_ahead * 10
        
        if estimated_minutes == 0:
            estimated_wait = "Ready soon"
        elif estimated_minutes <= 15:
            estimated_wait = f"{estimated_minutes} minutes"
        else:
            estimated_wait = f"{estimated_minutes//10*10}-{(estimated_minutes//10+1)*10} minutes"
        
        return {
            "success": True,
            "position_info": {
                "current_position": order["position"],
                "total_in_queue": total_in_queue,
                "orders_ahead": orders_ahead,
                "estimated_wait": estimated_wait
            },
            "order_state": order["state"],
            "order_reference": order["name"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving order position: {str(e)}"
        }


# ==== NEW MCP TOOL DESCRIPTION ====

GET_SERVICE_QUEUE_DESCRIPTION = """
Get the current queue status for a specific fast service. This tool provides real-time 
information about orders in the queue, helping staff and customers understand current workload.

## When to Use This Tool

Use this tool when:
1. **Queue Monitoring**: When staff need to see current workload
2. **Customer Information**: When customers want to know queue length before ordering
3. **Service Management**: When managers need to assess service capacity
4. **Wait Time Estimation**: When providing realistic wait time estimates
5. **Load Balancing**: When directing customers to less busy services

## Parameters

- **service_name**: The name of the service to check queue for

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **service_info**: Service details (name, type, queue count)
- **queue_details**: Detailed queue information including:
  - total_in_queue: Current number of orders in queue
  - confirmed_orders: Orders waiting to be prepared
  - preparing_orders: Orders currently being prepared
  - average_wait_estimate: Estimated wait time for new orders
- **queue_orders**: List of orders currently in queue with basic details
- **error**: Error message if the operation failed

## Example Usage

```python
result = get_service_queue("Dulayni")
# Returns: {
#   "success": True,
#   "service_info": {
#     "name": "Dulayni",
#     "type": "other",
#     "queue_count": 5
#   },
#   "queue_details": {
#     "total_in_queue": 5,
#     "confirmed_orders": 3,
#     "preparing_orders": 2,
#     "average_wait_estimate": "25-30 minutes"
#   },
#   "queue_orders": [...]
# }
```
"""

# ==== NEW MCP TOOL ====

@mcp.tool(description=GET_SERVICE_QUEUE_DESCRIPTION)
def get_service_queue(
    ctx: Context,
    service_name: str,
) -> Dict[str, Any]:
    """
    Get the current queue status for a specific fast service
    
    Parameters:
        service_name: The name of the service
        
    Returns:
        Dictionary containing queue information or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Find the service
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {"fields": ["id", "name", "service_type", "queue"], "limit": 1}
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service = service_result[0]
        service_id = service["id"]
        
        # Get detailed queue information
        confirmed_count = odoo.execute_method(
            "fast_service.order",
            "search_count",
            [
                ("service_id", "=", service_id),
                ("state", "=", "confirmed")
            ]
        )
        
        preparing_count = odoo.execute_method(
            "fast_service.order",
            "search_count",
            [
                ("service_id", "=", service_id),
                ("state", "=", "preparing")
            ]
        )
        
        # Get queue orders with basic details
        queue_orders_result = odoo.execute_method(
            "fast_service.order",
            "search_read",
            [
                ("service_id", "=", service_id),
                ("state", "in", ["confirmed", "preparing"])
            ],
            {
                "fields": ["id", "name", "customer_id", "state", "position", "total", "date"],
                "order": "date asc"
            }
        )
        
        # Format queue orders
        queue_orders = [
            {
                "id": order["id"],
                "reference": order["name"],
                "customer_name": order["customer_id"][1] if order["customer_id"] else "Unknown",
                "state": order["state"],
                "position": order["position"],
                "total": order["total"],
                "order_time": order["date"]
            }
            for order in queue_orders_result
        ]
        
        # Calculate estimated wait time
        total_queue = confirmed_count + preparing_count
        # Rough estimate: 5 minutes per confirmed order, 2 minutes remaining for preparing orders
        estimated_minutes = (confirmed_count * 5) + (preparing_count * 2)
        
        if estimated_minutes == 0:
            wait_estimate = "No wait - service available"
        elif estimated_minutes <= 10:
            wait_estimate = f"{estimated_minutes} minutes"
        elif estimated_minutes <= 30:
            wait_estimate = f"{estimated_minutes//5*5}-{(estimated_minutes//5+1)*5} minutes"
        else:
            wait_estimate = f"{estimated_minutes//10*10}-{(estimated_minutes//10+1)*10} minutes"
        
        return {
            "success": True,
            "service_info": {
                "id": service["id"],
                "name": service["name"],
                "type": service["service_type"],
                "queue_count": service["queue"]
            },
            "queue_details": {
                "total_in_queue": total_queue,
                "confirmed_orders": confirmed_count,
                "preparing_orders": preparing_count,
                "average_wait_estimate": wait_estimate
            },
            "queue_orders": queue_orders
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving service queue: {str(e)}"
        }

# ==== NEW MCP TOOL DESCRIPTION ====

GET_SERVICE_PERFORMANCE_DESCRIPTION = """
Get performance metrics and timing statistics for a specific fast service. This tool provides 
comprehensive analytics about service efficiency, completion times, and operational insights.

## When to Use This Tool

Use this tool when:
1. **Performance Analysis**: When managers need service efficiency metrics
2. **Capacity Planning**: When determining optimal staffing and resources
3. **Customer Communication**: When providing accurate wait time estimates
4. **Quality Monitoring**: When tracking service improvements over time
5. **Benchmarking**: When comparing performance across different services
6. **Operational Optimization**: When identifying bottlenecks and improvement areas

## Parameters

- **service_name**: The name of the service to analyze

## Returns

Dictionary containing:
- **success**: Boolean indicating if the operation was successful
- **service_info**: Basic service information
- **performance_metrics**: Detailed timing statistics including:
  - min_completion_time: Fastest order completion time
  - max_completion_time: Slowest order completion time  
  - avg_completion_time: Average completion time
  - total_completed_orders: Number of orders with timing data
- **current_status**: Current queue and operational status
- **recent_performance**: Performance metrics for recent orders (last 30 days)
- **error**: Error message if the operation failed

## Example Usage

```python
result = get_service_performance("Dulayni")
# Returns: {
#   "success": True,
#   "service_info": {
#     "name": "Dulayni",
#     "type": "other"
#   },
#   "performance_metrics": {
#     "min_completion_time": 5.2,
#     "max_completion_time": 45.8,
#     "avg_completion_time": 18.3,
#     "total_completed_orders": 156
#   },
#   "current_status": {
#     "queue": 3,
#     "estimated_wait": "25-30 minutes"
#   }
# }
```
"""

# ==== NEW MCP TOOL ====

@mcp.tool(description=GET_SERVICE_PERFORMANCE_DESCRIPTION)
def get_service_performance(
    ctx: Context,
    service_name: str,
) -> Dict[str, Any]:
    """
    Get performance metrics and timing statistics for a specific service
    
    Parameters:
        service_name: The name of the service
        
    Returns:
        Dictionary containing performance metrics or error message
    """
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Find the service
        service_result = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [("name", "ilike", service_name)],
            {
                "fields": [
                    "id", "name", "service_type", "queue",
                    "min_completion_time", "max_completion_time", 
                    "avg_completion_time", "total_completed_orders"
                ],
                "limit": 1
            }
        )
        
        if not service_result:
            return {
                "success": False,
                "error": f"Service '{service_name}' not found"
            }
        
        service = service_result[0]
        service_id = service["id"]
        
        # Get recent performance (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        
        recent_orders = odoo.execute_method(
            "fast_service.order",
            "search_read",
            [
                ("service_id", "=", service_id),
                ("state", "=", "delivered"),
                ("end_time", ">=", thirty_days_ago),
                ("start_time", "!=", False),
                ("end_time", "!=", False)
            ],
            {"fields": ["completion_time"]}
        )
        
        # Calculate recent performance metrics
        recent_completion_times = [order["completion_time"] for order in recent_orders if order["completion_time"] > 0]
        
        if recent_completion_times:
            recent_min = min(recent_completion_times)
            recent_max = max(recent_completion_times)
            recent_avg = sum(recent_completion_times) / len(recent_completion_times)
        else:
            recent_min = recent_max = recent_avg = 0.0
        
        # Calculate estimated wait time based on current queue and average completion time
        current_queue = service["queue"]
        avg_time = service["avg_completion_time"] if service["avg_completion_time"] > 0 else 15  # Default 15 min
        estimated_wait_minutes = current_queue * avg_time
        
        if estimated_wait_minutes == 0:
            estimated_wait = "No wait - service available"
        elif estimated_wait_minutes <= 15:
            estimated_wait = f"{int(estimated_wait_minutes)} minutes"
        else:
            estimated_wait = f"{int(estimated_wait_minutes//10*10)}-{int((estimated_wait_minutes//10+1)*10)} minutes"
        
        return {
            "success": True,
            "service_info": {
                "id": service["id"],
                "name": service["name"],
                "type": service["service_type"]
            },
            "performance_metrics": {
                "min_completion_time": round(service["min_completion_time"], 1),
                "max_completion_time": round(service["max_completion_time"], 1),
                "avg_completion_time": round(service["avg_completion_time"], 1),
                "total_completed_orders": service["total_completed_orders"]
            },
            "current_status": {
                "queue": current_queue,
                "estimated_wait": estimated_wait
            },
            "recent_performance": {
                "last_30_days": {
                    "min_completion_time": round(recent_min, 1),
                    "max_completion_time": round(recent_max, 1),
                    "avg_completion_time": round(recent_avg, 1),
                    "total_orders": len(recent_completion_times)
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving service performance: {str(e)}"
        }
