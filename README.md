# 2. Define State Schema and widgets (if on Databricks)
class InvoiceState(TypedDict):
    input_path: str
    extract_all: bool
    extract_invoice_amount: bool
    extract_itemise: bool
    invoice_data: Optional[Any]
    total_invoices_processed: Optional[int]
    average_time_per_invoice: Optional[float]
    node_details: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    final_response: Optional[Dict[str, Any]]

# Databricks widgets (skip if not on Databricks)
try:
    dbutils.widgets.text("input_path", "", "Input Path")
    dbutils.widgets.dropdown("extract_all", "true", ["true", "false"], "Extract All")
    dbutils.widgets.dropdown("extract_invoice_amount", "false", ["true", "false"], "Extract Invoice Amount")
    dbutils.widgets.dropdown("extract_itemise", "false", ["true", "false"], "Extract Itemise")
    input_path = dbutils.widgets.get("input_path")
    extract_all = dbutils.widgets.get("extract_all") == "true"
    extract_invoice_amount = dbutils.widgets.get("extract_invoice_amount") == "true"
    extract_itemise = dbutils.widgets.get("extract_itemise") == "true"
except Exception:
    # Fallbacks for local runs
    input_path = "/path/to/your/invoices"
    extract_all = True
    extract_invoice_amount = False
    extract_itemise = False
