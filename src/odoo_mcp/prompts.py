# Step 1: Add this description constant at the top of your Odoo MCP server file
EXECUTE_METHOD_DESCRIPTION = """Execute custom methods on Odoo models to interact with the ERP system. This tool provides direct access to Odoo's ORM methods, enabling complex data operations, business logic execution, and system interactions.

## When to Use This Tool

Use this tool proactively in these scenarios:

1. **Complex Data Operations**: When you need to perform operations beyond simple CRUD (Create, Read, Update, Delete)
2. **Business Logic Execution**: When you need to trigger Odoo's built-in business methods or workflows
3. **Batch Operations**: When you need to process multiple records efficiently
4. **Advanced Queries**: When simple search operations aren't sufficient and you need custom filtering or aggregation
5. **System Integration**: When you need to interact with Odoo's internal mechanisms or trigger automated processes
6. **Multi-step Workflows**: When implementing complex business processes that span multiple models

## When NOT to Use This Tool

Skip using this tool when:
1. Simple record lookup by ID is sufficient (use get_record resource instead)
2. Basic search operations can be handled by search_records resource
3. You just need to list available models (use get_models resource)
4. You only need model field information (use get_model_info resource)

## Common Odoo Methods and Their Use Cases

### Data Retrieval Methods
- `search(domain, limit, offset, order)`: Find record IDs matching criteria
- `search_read(domain, fields, limit, offset, order)`: Search and read in one operation
- `read(ids, fields)`: Read specific fields from records by ID
- `browse(ids)`: Get record objects for further manipulation

### Data Modification Methods
- `create(vals_list)`: Create new records
- `write(vals)`: Update existing records
- `unlink()`: Delete records
- `copy(default)`: Duplicate records with modifications

### Business Logic Methods
- `action_confirm()`: Confirm documents (sales orders, invoices, etc.)
- `action_cancel()`: Cancel documents
- `action_done()`: Mark processes as complete
- `compute_*()`: Trigger field computations
- `_compute_*()`: Execute specific computation methods

### Utility Methods
- `name_search(name, domain, operator, limit)`: Search by name with fuzzy matching
- `name_get()`: Get display names for records
- `fields_get(fields, attributes)`: Get field definitions and metadata
- `default_get(fields)`: Get default values for new records

## Parameter Guidelines

### Model Parameter
- Use the technical model name (e.g., 'res.partner', 'sale.order', 'hr.employee')
- Common models: 'res.partner' (contacts), 'sale.order' (sales), 'purchase.order' (purchases), 'hr.employee' (employees)

### Args Parameter (Positional Arguments)
- Most methods expect a list of record IDs as the first argument when operating on existing records
- For search methods, the first argument is typically the domain (search criteria)
- Example: `[1, 2, 3]` for record IDs, `[['name', 'ilike', 'John']]` for domain

### Kwargs Parameter (Keyword Arguments)
- Used for method-specific parameters like 'limit', 'offset', 'fields', 'order'
- For create/write operations, use 'vals' or 'vals_list' to pass record data
- Example: `{"limit": 10, "offset": 0, "fields": ["name", "email"]}`

## Domain Construction for Search Operations

Domains use Polish notation with conditions in the format: `[field, operator, value]`

### Basic Operators
- `'='`: Exact match
- `'!='`: Not equal
- `'>'`, `'>='`, `'<'`, `'<='`: Numeric/date comparisons
- `'in'`: Value in list
- `'not in'`: Value not in list
- `'like'`: Case-sensitive pattern matching (use % for wildcards)
- `'ilike'`: Case-insensitive pattern matching
- `'=like'`: Case-sensitive exact pattern
- `'=ilike'`: Case-insensitive exact pattern

### Logical Operators
- `'&'`: AND (default between conditions)
- `'|'`: OR
- `'!'`: NOT

### Domain Examples
```python
# Simple condition
[['name', 'ilike', 'john']]

# Multiple conditions (AND by default)
[['name', 'ilike', 'john'], ['active', '=', True]]

# OR condition
['|', ['name', 'ilike', 'john'], ['email', 'ilike', 'john']]

# Complex condition with NOT
['!', ['state', 'in', ['draft', 'cancel']]]

# Date range
[['create_date', '>=', '2024-01-01'], ['create_date', '<=', '2024-12-31']]
```

## Multi-Step Task Solving Examples

### Example 1: Customer Order Analysis
```python
# Step 1: Find active customers
execute_method(
    model="res.partner",
    method="search_read",
    args=[[['customer_rank', '>', 0], ['active', '=', True]]],
    kwargs={"fields": ["name", "id"], "limit": 50}
)

# Step 2: Get their recent orders
execute_method(
    model="sale.order",
    method="search_read",
    args=[[['partner_id', 'in', [1,2,3]], ['create_date', '>=', '2024-01-01']]],
    kwargs={"fields": ["name", "amount_total", "state", "partner_id"]}
)

# Step 3: Calculate order statistics
execute_method(
    model="sale.order",
    method="read_group",
    args=[[['state', '=', 'sale']]],
    kwargs={
        "fields": ["amount_total:sum"],
        "groupby": ["partner_id"]
    }
)
```

### Example 2: Employee Holiday Management
```python
# Step 1: Find employees in specific department
execute_method(
    model="hr.employee",
    method="search",
    args=[[['department_id.name', '=', 'IT Department']]],
    kwargs={"limit": 100}
)

# Step 2: Check their holiday balances
execute_method(
    model="hr.leave.allocation",
    method="search_read",
    args=[[['employee_id', 'in', employee_ids], ['state', '=', 'validate']]],
    kwargs={"fields": ["employee_id", "number_of_days", "holiday_status_id"]}
)

# Step 3: Create new holiday request
execute_method(
    model="hr.leave",
    method="create",
    kwargs={
        "vals": {
            "employee_id": 1,
            "holiday_status_id": 1,
            "request_date_from": "2024-12-25",
            "request_date_to": "2024-12-25",
            "number_of_days": 1
        }
    }
)
```

### Example 3: Inventory Management Workflow
```python
# Step 1: Check current stock levels
execute_method(
    model="stock.quant",
    method="search_read",
    args=[[['product_id.name', 'ilike', 'laptop'], ['location_id.usage', '=', 'internal']]],
    kwargs={"fields": ["product_id", "quantity", "location_id"]}
)

# Step 2: Create stock adjustment if needed
execute_method(
    model="stock.inventory",
    method="create",
    kwargs={
        "vals": {
            "name": "Stock Adjustment - Laptops",
            "location_ids": [(6, 0, [location_id])]
        }
    }
)

# Step 3: Confirm the inventory adjustment
execute_method(
    model="stock.inventory",
    method="action_start",
    args=[inventory_id]
)
```

## Error Handling and Troubleshooting

### Common Error Scenarios
1. **Access Denied**: User lacks permissions for the model/method
2. **Validation Error**: Data doesn't meet Odoo's validation rules
3. **Foreign Key Error**: Referenced records don't exist
4. **Method Not Found**: Method doesn't exist on the model
5. **Invalid Domain**: Search domain is malformed

### Best Practices
1. **Start Simple**: Begin with read operations before attempting modifications
2. **Validate References**: Ensure referenced records exist before creating relationships
3. **Check Permissions**: Verify user has appropriate access rights
4. **Use Transactions**: Group related operations when possible
5. **Handle Responses**: Always check the success/error fields in responses

## Advanced Usage Patterns

### Working with Relational Fields
```python
# Many2one: Use integer ID
{"partner_id": 123}

# One2many/Many2many: Use command tuples
{"line_ids": [(0, 0, {"product_id": 1, "quantity": 5})]}  # Create new line
{"tag_ids": [(6, 0, [1, 2, 3])]}  # Replace all tags with these IDs
{"line_ids": [(1, line_id, {"quantity": 10})]}  # Update existing line
{"line_ids": [(2, line_id)]}  # Delete line
```

### Bulk Operations
```python
# Create multiple records at once
execute_method(
    model="res.partner",
    method="create",
    kwargs={
        "vals_list": [
            {"name": "Customer A", "email": "a@example.com"},
            {"name": "Customer B", "email": "b@example.com"}
        ]
    }
)

# Update multiple records
execute_method(
    model="res.partner",
    method="write",
    args=[record_ids],
    kwargs={"vals": {"active": False}}
)
```

### Custom Business Methods
```python
# Trigger specific business logic
execute_method(
    model="sale.order",
    method="action_confirm",
    args=[order_id]
)

# Execute custom methods with complex parameters
execute_method(
    model="account.invoice",
    method="action_post",
    args=[invoice_ids],
    kwargs={"force_post": True}
)
```

## Response Format

The tool returns a dictionary with:
- `success` (bool): Whether the operation succeeded
- `result` (any): The actual result from Odoo (if successful)
- `error` (str): Error message (if failed)

Always check the `success` field before processing results. The `result` field contains the raw response from Odoo and may be a list, dictionary, boolean, or other data type depending on the method called."""
