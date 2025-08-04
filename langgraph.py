# Databricks notebook source

# --- 1. Import and Install ---
# (Use only necessary packages for clarity)
# MAGIC %pip install torch torchvision transformers pillow langgraph

import os
import json
import time
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END

# --- 2. Define State Schema ---
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

# --- 3. Widgets for Databricks (or fallback) ---
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
    input_path = "/path/to/invoice/folder"
    extract_all = True
    extract_invoice_amount = False
    extract_itemise = False

# --- 4. User Input Node ---
def user_input_node(state: InvoiceState) -> InvoiceState:
    input_path = dbutils.widgets.get("input_path")
    extract_all = dbutils.widgets.get("extract_all") == "true"
    extract_invoice_amount = dbutils.widgets.get("extract_invoice_amount") == "true"
    extract_itemise = dbutils.widgets.get("extract_itemise") == "true"
    state["input_path"] = input_path
    state["extract_all"] = extract_all
    state["extract_invoice_amount"] = extract_invoice_amount
    state["extract_itemise"] = extract_itemise
    return state

# --- 5. InvoiceAnalyzer and Processing Functions (STUB for demo, replace with real logic) ---
class InvoiceAnalyzer:
    def __init__(self, input_path: str):
        self.input_path = input_path
        # Add prompt_template if needed for your model

    def extract_all(self, json_data: Dict) -> Dict:
        return json_data.get("invoice_data", {})

    def extract_itemize(self, json_data: Dict) -> Dict:
        return json_data.get("invoice_data", {}).get("table", {})

    def extract_invoice_amount(self, json_data: Dict) -> Optional[float]:
        # Dummy example
        return json_data.get("invoice_data", {}).get("total", None)

def process_images_stub(image_folder, output_folder, batch_size=2):
    # Dummy stub for demo
    os.makedirs(output_folder, exist_ok=True)
    dummy_json = {
        "invoice_data": {
            "table": {"items": [{"item": "A", "qty": 1}]},
            "total": 100.0,
            "non_table_text": {"InvoiceNumber": "1234"}
        }
    }
    for i in range(3):
        with open(os.path.join(output_folder, f"invoice_{i}.json"), "w") as f:
            json.dump(dummy_json, f)
    with open(os.path.join(output_folder, "processing_stats.json"), "w") as f:
        json.dump({"total_images_processed": 3, "average_processing_time_per_image": 1.0}, f)

# --- 6. Node Functions ---
def extract_all_node(state: InvoiceState) -> InvoiceState:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/tmp/invoice_output"
    process_images_stub(image_folder, output_folder)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json') and f != "processing_stats.json"]

    all_data, tabular_count, invoice_count, itemise_count = [], 0, 0, 0
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            res = analyzer.extract_all(json_data)
            all_data.append(res)
            if "table" in res:
                tabular_count += 1
            if analyzer.extract_invoice_amount(json_data) is not None:
                invoice_count += 1
            if "table" in res:
                itemise_count += 1
    total_time = time.time() - start_time
    total_count = len(json_files)
    node_details = [
        {"node": "tabular", "processed_count": tabular_count, "average_time": round(total_time/total_count, 2) if total_count else 0},
        {"node": "invoice_amount", "processed_count": invoice_count, "average_time": round(total_time/total_count, 2) if total_count else 0},
        {"node": "itemise", "processed_count": itemise_count, "average_time": round(total_time/total_count, 2) if total_count else 0},
    ]
    state["invoice_data"] = all_data
    state["total_invoices_processed"] = total_count
    state["average_time_per_invoice"] = round(total_time/total_count, 2) if total_count else 0
    state["node_details"] = node_details
    return state

def extract_invoice_amount_node(state: InvoiceState) -> InvoiceState:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/tmp/invoice_output"
    process_images_stub(image_folder, output_folder)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json') and f != "processing_stats.json"]

    invoice_amounts = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            amount = analyzer.extract_invoice_amount(json_data)
            if amount is not None:
                invoice_amounts.append(amount)
    total_time = time.time() - start_time
    count = len(invoice_amounts)
    state["invoice_amount"] = invoice_amounts
    state["total_invoices_processed"] = count
    state["average_time_per_invoice"] = round(total_time / count, 2) if count else 0
    return state

def extract_itemize_node(state: InvoiceState) -> InvoiceState:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/tmp/invoice_output"
    process_images_stub(image_folder, output_folder)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json') and f != "processing_stats.json"]

    itemized_data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            itemized_data.append(analyzer.extract_itemize(json_data))
    total_time = time.time() - start_time
    count = len(itemized_data)
    state["items_table"] = itemized_data
    state["total_invoices_processed"] = count
    state["average_time_per_invoice"] = round(total_time / count, 2) if count else 0
    return state

def result_node(state: InvoiceState) -> InvoiceState:
    # Always output dict!
    state['final_response'] = {
        "processed_count": state.get("total_invoices_processed", 0),
        "average_processing_time_per_invoice": state.get("average_time_per_invoice", 0),
        "node_details": state.get("node_details", None)
    }
    return state

# --- 7. LangGraph Workflow ---
workflow = StateGraph(InvoiceState)
workflow.add_node("user_input_node", user_input_node)
workflow.add_node("extract_all_node", extract_all_node)
workflow.add_node("extract_invoice_amount_node", extract_invoice_amount_node)
workflow.add_node("extract_itemize_node", extract_itemize_node)
workflow.add_node("result_node", result_node)
workflow.set_entry_point("user_input_node")

def route_based_on_user_selection(state: InvoiceState) -> str:
    if state.get("extract_all"):
        return "extract_all_node"
    elif state.get("extract_invoice_amount"):
        return "extract_invoice_amount_node"
    elif state.get("extract_itemise"):
        return "extract_itemize_node"
    else:
        return "result_node"

workflow.add_conditional_edges(
    "user_input_node",
    route_based_on_user_selection,
    {
        "extract_all_node": "extract_all_node",
        "extract_invoice_amount_node": "extract_invoice_amount_node",
        "extract_itemize_node": "extract_itemize_node",
        "result_node": "result_node"
    }
)
workflow.add_edge("extract_all_node", "result_node")
workflow.add_edge("extract_invoice_amount_node", "result_node")
workflow.add_edge("extract_itemize_node", "result_node")
workflow.add_edge("result_node", END)

invoice_workflow = workflow.compile()

# --- 8. Running the Workflow & Return Block ---
initial_state = { 
    "input_path": input_path, 
    "extract_all": extract_all, 
    "extract_invoice_amount": extract_invoice_amount, 
    "extract_itemise": extract_itemise 
}
final_state = invoice_workflow.invoke(initial_state)

# Defensive: always expect dict, never call `.get` on a string.
final_result = final_state['final_response'] if isinstance(final_state.get('final_response', None), dict) else {
    "processed_count": 0,
    "average_processing_time_per_invoice": 0,
    "node_details": None
}

result_data = { 
    "processed_count": final_result['processed_count'], 
    "average_processing_time_per_invoice": final_result['average_processing_time_per_invoice'], 
    "node_details": final_result['node_details'] 
}

try:
    dbutils.notebook.exit(json.dumps({"status": "success", "result": result_data}))
except Exception:
    print(json.dumps({"status": "success", "result": result_data}))