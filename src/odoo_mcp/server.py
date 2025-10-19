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


# ============================================================================
# CUSTOMER FLOW TOOLS
# ============================================================================

@mcp.tool(description="Show service menu to customer via WhatsApp list message")
def show_service_menu(
    ctx: Context,
    target_phone: str,
    sender_phone: str,
    service_name: str
) -> Dict:
    """Display a service's product menu to a customer"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Look up the customer by phone number
        partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["phone", "ilike", target_phone]],
            ["id"]
        )
        
        if not partners:
            return {"status": "partner_not_found", "message": f"Customer with phone {target_phone} not found"}
        
        partner_id = partners[0]["id"]
        
        # Search for service
        services = odoo.execute_method(
            "fast_service.service",
            "search_read",
            [["name", "ilike", service_name]],
        )
        
        if not services:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [partner_id],
                sender_phone,
                f"❌ Service « {service_name} » introuvable.\nVeuillez vérifier le nom et réessayer."
            )
            return {"status": "not_found", "message": f"Service {service_name} not found"}
        
        service = services[0]
        product_ids = service["product_ids"]
        
        if not product_ids:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [partner_id],
                sender_phone,
                f"❌ Aucun produit disponible pour {service['name']}"
            )
            return {"status": "no_products", "message": "No products available"}
        
        # Get products
        products = odoo.execute_method(
            "product.product",
            "read",
            product_ids,
        )
        
        # Build list message
        rows = []
        for product in products[:10]:
            rows.append({
                "id": f"product_{service['id']}_{product['id']}",
                "title": str(product["name"])[:24],
                "description": f"{int(product['list_price'])} XOF"
            })
        
        sections = [{"title": "Menu", "rows": rows}]
        
        # Send list message
        odoo.execute_method(
            "res.partner",
            "send_list_message",
            [partner_id],
            sender_phone,
            f"Bienvenue chez {service['name']} ! 🎉\n\nSélectionnez vos articles :",
            "Voir le menu",
            sections,
            footer_text="Fast Service"
        )
        
        return {
            "status": "success",
            "partner_id": partner_id,
            "service_id": service["id"],
            "service_name": service["name"],
            "products_shown": len(rows)
        }
    
    except Exception as e:
        logger.exception(f"Error showing menu: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Add product to customer's cart")
def add_to_cart(
    ctx: Context,
    phone: str,
    service_id: int,
    product_id: int,
    sender_phone: str
) -> Dict:
    """Add a product to customer's order (draft order acts as cart)"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get or create partner
        partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
        )
        
        if not partners:
            partner_id = odoo.execute_method(
                "res.partner",
                "create",
                [{"name": phone, "mobile": phone, "phone": phone}]
            )
        else:
            partner_id = partners[0]["id"]
        
        # Get product
        products = odoo.execute_method(
            "product.product",
            "read",
            [product_id],
        )
        product = products[0]
        
        # Check for existing draft order
        orders = odoo.execute_method(
            "fast_service.order",
            "search_read",
            [
                ["customer_id", "=", partner_id],
                ["service_id", "=", service_id],
                ["state", "=", "draft"]
            ],
        )
        
        if orders:
            order_id = orders[0]["id"]
            # Add line
            odoo.execute_method(
                "fast_service.order.line",
                "create",
                [{
                    "order_id": order_id,
                    "product_id": product_id,
                    "quantity": 1,
                    "price": product["list_price"]
                }]
            )
        else:
            # Create new order
            order_id = odoo.execute_method(
                "fast_service.order",
                "create",
                [{
                    "customer_id": partner_id,
                    "service_id": service_id,
                    "state": "draft",
                    "order_line_ids": [[0, 0, {
                        "product_id": product_id,
                        "quantity": 1,
                        "price": product["list_price"]
                    }]]
                }]
            )
        
        # Get updated order
        orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
        )
        order = orders[0]
        
        # Get lines
        lines = odoo.execute_method(
            "fast_service.order.line",
            "read",
            order["order_line_ids"],
        )
        
        items_text = "\n".join([
            f"• {line['quantity']}x {line['product_id'][1]} - {int(line['price'])} XOF"
            for line in lines
        ])
        
        # Send confirmation
        odoo.execute_method(
            "res.partner",
            "send_button_message",
            [partner_id],
            sender_phone,
            f"📋 Votre commande :\n{items_text}\n\n💰 Total : {int(order['total'])} XOF\n\nConfirmez-vous cette commande ?",
            [
                {"id": f"confirm_order_{order_id}", "title": "✅ Confirmer"},
                {"id": f"cancel_order_{order_id}", "title": "❌ Annuler"}
            ],
            footer_text=f"Commande {order['name']}"
        )
        
        return {
            "status": "success",
            "order_id": order_id,
            "order_name": order["name"],
            "total": order["total"],
            "item_count": len(lines)
        }
        
    except Exception as e:
        logger.exception(f"Error adding to cart: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Confirm customer order and initiate payment")
def confirm_order(
    ctx: Context,
    phone: str,
    order_id: int,
    sender_phone: str
) -> Dict:
    """Confirm a customer's order and send payment link"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not partners:
            return {"status": "partner_not_found", "message": f"Customer with phone {phone} not found"}
        
        partner_id = partners[0]["id"]
        
        # Get order
        orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
        )
        order = orders[0]
        
        if order["state"] != "draft":
            odoo.execute_method(
                "res.partner",
                "send_message",
                [partner_id],
                sender_phone,
                "❌ Commande déjà traitée"
            )
            return {"status": "invalid_state", "message": "Order already processed"}
        
        # Check minimum topup
        # services = odoo.execute_method(
        #     "fast_service.service",
        #     "read",
        #     [order["service_id"][0]],
        # )
        # service = services[0]
        
        # if order["total"] < service["minimum_topup"]:
        #     odoo.execute_method(
        #         "res.partner",
        #         "send_message",
        #         [partner_id],
        #         sender_phone,
        #         f"❌ Montant minimum : {int(service['minimum_topup'])} XOF"
        #     )
        #     return {"status": "below_minimum", "message": "Below minimum amount"}
        
        # Update to confirmed
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order_id],
            {"state": "confirmed"}
        )
        
        # Get Wave config
        wave_configs = odoo.execute_method(
            "wave.service",
            "search_read",
            [],
        )
        
        if not wave_configs:
            return {"status": "error", "message": "Wave config not found"}
        
        wave_config_id = wave_configs[0]["id"]
        
        # Create payment
        payment = odoo.execute_method(
            "wave.service",
            "create_checkout_session",
            [wave_config_id],
            phone,
            int(order["total"])
        )
        
        payment_url = payment["wave_launch_url"]
        
        # Send payment link
        odoo.execute_method(
            "res.partner",
            "send_message",
            [partner_id],
            sender_phone,
            f"💳 Montant à payer : {int(order['total'])} XOF\n\n"
            f"Cliquez sur le lien pour payer via Wave :\n{payment_url}"
        )
        
        # Get queue position
        updated_orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
        )
        updated_order = updated_orders[0]
        
        if updated_order["position"] > 0:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [partner_id],
                sender_phone,
                f"⏳ Votre position dans la file : #{updated_order['position']}\n\n"
                "Nous vous informerons dès que votre commande sera en préparation."
            )
        
        return {
            "status": "success",
            "order_id": order_id,
            "payment_url": payment_url,
            "queue_position": updated_order["position"]
        }
        
    except Exception as e:
        logger.exception(f"Error confirming order: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Cancel a customer order")
def cancel_order(
    ctx: Context,
    phone: str,
    order_id: int,
    sender_phone: str
) -> Dict:
    """Cancel a customer's draft order"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not partners:
            return {"status": "partner_not_found", "message": f"Customer with phone {phone} not found"}
        
        partner_id = partners[0]["id"]
        
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order_id],
            {"state": "canceled"}
        )
        
        odoo.execute_method(
            "res.partner",
            "send_message",
            [partner_id],
            sender_phone,
            "❌ Commande annulée"
        )
        
        return {"status": "success", "order_id": order_id}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# STAFF (PREPARER) FLOW TOOLS
# ============================================================================

@mcp.tool(description="Get next order for staff preparer")
def get_next_order_for_preparer(
    ctx: Context,
    phone: str,
    sender_phone: str
) -> Dict:
    """Get the next order in queue for a preparer and mark it as preparing"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        preparer_partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not preparer_partners:
            return {"status": "partner_not_found", "message": f"Preparer with phone {phone} not found"}
        
        preparer_partner_id = preparer_partners[0]["id"]
        
        # Find service
        services = odoo.execute_method(
            "fast_service.service",
            "search_read",
            ["|", ["preparer_id.mobile", "=", phone], ["deliverer_id.mobile", "=", phone]],
        )
        
        if not services:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [preparer_partner_id],
                sender_phone,
                "⚠️ Vous n'êtes affecté à aucun service.\nVeuillez contacter votre administrateur."
            )
            return {"status": "no_service"}
        
        service = services[0]
        
        # Get next order
        orders = odoo.execute_method(
            "fast_service.order",
            "search_read",
            [["service_id", "=", service["id"]], ["state", "=", "confirmed"]],
            {"fields": ["id", "name", "total", "order_line_ids", "customer_id"], "order": "date asc", "limit": 1}
        )
        
        if not orders:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [preparer_partner_id],
                sender_phone,
                "✅ Aucune commande en attente pour le moment."
            )
            return {"status": "no_orders"}
        
        order = orders[0]
        
        # Get lines
        lines = odoo.execute_method(
            "fast_service.order.line",
            "read",
            order["order_line_ids"],
            {"fields": ["product_id", "quantity"]}
        )
        
        items_text = "\n".join([f"• {line['quantity']}x {line['product_id'][1]}" for line in lines])
        
        # Send to preparer
        odoo.execute_method(
            "res.partner",
            "send_button_message",
            [preparer_partner_id],
            sender_phone,
            f"🔔 Nouvelle commande #{order['name']} :\n\n{items_text}\n\n💰 Total : {int(order['total'])} XOF",
            [
                {"id": f"complete_{order['id']}", "title": "✅ Terminée"},
                {"id": f"skip_{order['id']}", "title": "⏭️ Passer"}
            ],
            header_text="Commande en attente"
        )
        
        # Update to preparing
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order["id"]],
            {"state": "preparing", "start_time": datetime.now().isoformat()}
        )
        
        # Notify customer
        customers = odoo.execute_method(
            "res.partner",
            "read",
            [order["customer_id"][0]],
            {"fields": ["mobile"]}
        )
        customer = customers[0]
        
        if customer["mobile"]:
            avg_time = service.get("avg_completion_time", 15)
            odoo.execute_method(
                "res.partner",
                "send_message",
                [order["customer_id"][0]],
                sender_phone,
                f"👨‍🍳 Votre commande #{order['name']} est en cours de préparation !\n\n"
                f"Temps d'attente estimé : {int(avg_time)} minutes"
            )
        
        return {
            "status": "success",
            "order_id": order["id"],
            "order_name": order["name"]
        }
        
    except Exception as e:
        logger.exception(f"Error getting next order: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Mark order as complete and ready for delivery")
def complete_order_preparation(
    ctx: Context,
    phone: str,
    order_id: int,
    sender_phone: str
) -> Dict:
    """Mark an order as ready for delivery and notify deliverer"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        preparer_partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not preparer_partners:
            return {"status": "partner_not_found", "message": f"Preparer with phone {phone} not found"}
        
        preparer_partner_id = preparer_partners[0]["id"]
        
        # Update to ready
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order_id],
            {"state": "ready", "end_time": datetime.now().isoformat()}
        )
        
        # Get order details
        orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
            {"fields": ["name", "service_id", "customer_id"]}
        )
        order = orders[0]
        
        # Get service and deliverer
        services = odoo.execute_method(
            "fast_service.service",
            "read",
            [order["service_id"][0]],
            {"fields": ["deliverer_id"]}
        )
        service = services[0]
        
        # Notify deliverer
        if service["deliverer_id"]:
            deliverers = odoo.execute_method(
                "res.users",
                "read",
                [service["deliverer_id"][0]],
                {"fields": ["partner_id"]}
            )
            deliverer = deliverers[0]
            
            deliverer_partners = odoo.execute_method(
                "res.partner",
                "read",
                [deliverer["partner_id"][0]],
                {"fields": ["mobile", "id"]}
            )
            deliverer_partner = deliverer_partners[0]
            
            customers = odoo.execute_method(
                "res.partner",
                "read",
                [order["customer_id"][0]],
                {"fields": ["name", "street", "mobile"]}
            )
            customer = customers[0]
            
            if deliverer_partner["mobile"]:
                odoo.execute_method(
                    "res.partner",
                    "send_button_message",
                    [deliverer_partner["id"]],
                    sender_phone,
                    f"🚚 Commande #{order['name']} prête pour livraison !\n\n"
                    f"Client : {customer['name'] or customer['mobile']}\n"
                    f"Adresse : {customer.get('street') or 'Non spécifiée'}",
                    [{"id": f"deliver_{order_id}", "title": "📦 Livrer"}],
                    header_text="Livraison"
                )
        
        # Check for next order
        next_orders = odoo.execute_method(
            "fast_service.order",
            "search_read",
            [["service_id", "=", order["service_id"][0]], ["state", "=", "confirmed"]],
            {"fields": ["id"], "order": "date asc", "limit": 1}
        )
        
        if next_orders:
            return {"status": "success", "has_next": True}
        else:
            odoo.execute_method(
                "res.partner",
                "send_message",
                [preparer_partner_id],
                sender_phone,
                "✅ Aucune commande en attente pour le moment."
            )
            return {"status": "success", "has_next": False}
        
    except Exception as e:
        logger.exception(f"Error completing order: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Skip order to end of queue")
def skip_order(
    ctx: Context,
    phone: str,
    order_id: int,
    sender_phone: str
) -> Dict:
    """Move an order to the end of the queue"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order_id],
            {"date": datetime.now().isoformat(), "state": "confirmed"}
        )
        
        return {"status": "success", "order_id": order_id}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# DELIVERER FLOW TOOLS
# ============================================================================

@mcp.tool(description="Mark order as delivered and request feedback")
def deliver_order(
    ctx: Context,
    phone: str,
    order_id: int,
    sender_phone: str
) -> Dict:
    """Mark an order as delivered and request customer feedback"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        deliverer_partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not deliverer_partners:
            return {"status": "partner_not_found", "message": f"Deliverer with phone {phone} not found"}
        
        deliverer_partner_id = deliverer_partners[0]["id"]
        
        # Update to delivered
        odoo.execute_method(
            "fast_service.order",
            "write",
            [order_id],
            {"state": "delivered"}
        )
        
        # Get order and customer
        orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
            {"fields": ["name", "customer_id"]}
        )
        order = orders[0]
        
        customers = odoo.execute_method(
            "res.partner",
            "read",
            [order["customer_id"][0]],
            {"fields": ["mobile", "id"]}
        )
        customer = customers[0]
        
        # Request feedback
        if customer["mobile"]:
            odoo.execute_method(
                "res.partner",
                "send_button_message",
                [customer["id"]],
                sender_phone,
                f"⭐ Comment évalueriez-vous votre expérience ?\nCommande #{order['name']}",
                [
                    {"id": f"rating_{order_id}_5", "title": "⭐⭐⭐⭐⭐"},
                    {"id": f"rating_{order_id}_4", "title": "⭐⭐⭐⭐"},
                    {"id": f"rating_{order_id}_3", "title": "⭐⭐⭐"}
                ]
            )
        
        odoo.execute_method(
            "res.partner",
            "send_message",
            [deliverer_partner_id],
            sender_phone,
            f"✅ Commande #{order['name']} marquée comme livrée"
        )
        
        return {"status": "success", "order_id": order_id}
        
    except Exception as e:
        logger.exception(f"Error delivering order: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool(description="Record customer feedback rating")
def record_feedback(
    ctx: Context,
    phone: str,
    order_id: int,
    rating: int,
    sender_phone: str
) -> Dict:
    """Record customer feedback for an order"""
    odoo = ctx.request_context.lifespan_context.odoo
    
    try:
        # Get partner
        customer_partners = odoo.execute_method(
            "res.partner",
            "search_read",
            [["mobile", "=", phone]],
            ["id"]
        )
        
        if not customer_partners:
            return {"status": "partner_not_found", "message": f"Customer with phone {phone} not found"}
        
        customer_partner_id = customer_partners[0]["id"]
        
        # Get order
        orders = odoo.execute_method(
            "fast_service.order",
            "read",
            [order_id],
            {"fields": ["preparer_id", "deliverer_id"]}
        )
        order = orders[0]
        
        # Create feedback
        odoo.execute_method(
            "fast_service.staff.feedback",
            "create",
            [{
                "order_id": order_id,
                "preparer_id": order.get("preparer_id") and order["preparer_id"][0] or False,
                "deliverer_id": order.get("deliverer_id") and order["deliverer_id"][0] or False,
                "rating": rating
            }]
        )
        
        odoo.execute_method(
            "res.partner",
            "send_message",
            [customer_partner_id],
            sender_phone,
            "🙏 Merci pour votre retour ! À très bientôt."
        )
        
        return {"status": "success", "order_id": order_id, "rating": rating}
        
    except Exception as e:
        logger.exception(f"Error recording feedback: {str(e)}")
        return {"status": "error", "message": str(e)}
