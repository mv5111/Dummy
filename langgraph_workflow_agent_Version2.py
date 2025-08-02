from langgraph.graph import StateGraph, END

# Assume InvoiceAnalyzer and process_images are imported as in your code

def extract_all_node(state: dict) -> dict:
    import time
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder)
                  if fname.endswith('.json') and fname != "processing_stats.json"]
    tabular_times, invoice_times, itemise_times = [], [], []
    tabular_count = invoice_count = itemise_count = 0

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            result = analyzer.extract_all(json_data)
            all_data.append(result)
            if result.get("items_table"):
                tabular_count += 1
            if analyzer.extract_invoice_amount(json_data) is not None:
                invoice_count += 1
            if result.get("items_table"):
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
    state["average_time_per_invoice"] = round(total_time / total_count, 2) if total_count else 0
    state["node_details"] = node_details
    return state

def extract_invoice_amount_node(state: dict) -> dict:
    import time
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder)
                  if fname.endswith('.json') and fname != "processing_stats.json"]

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

def extract_itemize_node(state: dict) -> dict:
    import time
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder)
                  if fname.endswith('.json') and fname != "processing_stats.json"]

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

def result_node(state: dict) -> dict:
    # Format response for API
    state['final_response'] = {
        "processed_count": state.get("total_invoices_processed", 0),
        "average_processing_time_per_invoice": state.get("average_time_per_invoice", 0),
        "node_details": state.get("node_details", None)
    }
    return state

def user_input_node(state: dict) -> dict:
    # This node just passes state through, assuming input_path and booleans are set
    return state

# LangGraph workflow
workflow = StateGraph(dict)
workflow.add_node("user_input_node", user_input_node)
workflow.add_node("extract_all_node", extract_all_node)
workflow.add_node("extract_invoice_amount_node", extract_invoice_amount_node)
workflow.add_node("extract_itemize_node", extract_itemize_node)
workflow.add_node("result_node", result_node)
workflow.set_entry_point("user_input_node")

def route_based_on_user_selection(state: dict) -> str:
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

# To invoke:
# initial_state = {"input_path": ..., "extract_all": ..., "extract_invoice_amount": ..., "extract_itemise": ...}
# final_state = invoice_workflow.invoke(initial_state)
# dbutils.notebook.exit(json.dumps({"status": "success", "result": final_state['final_response']}))