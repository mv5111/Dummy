def robust_json_load(json_file):
    import json
    with open(json_file, 'r') as file:
        try:
            data = json.load(file)
            # If the data is a stringified JSON (double-encoded), decode again
            if isinstance(data, str):
                data = json.loads(data)
            return data
        except Exception as e:
            print(f"ERROR: Failed to load JSON from {json_file}: {e}")
            return None

def extract_all_node(state: dict) -> dict:
    import os, time
    print("DEBUG: Starting extract_all_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json')]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}")

    all_results = []
    for json_file in json_files:
        json_data = robust_json_load(json_file)
        if not json_data:
            print(f"WARNING: Skipping {json_file} due to load error.")
            continue
        # Use directly if already in expected format
        if "invoice_data" in json_data:
            all_results.append(json_data)
        else:
            # Fallback: wrap with analyzer if not in expected format
            all_results.append({"invoice_data": analyzer.extract_all(json_data)})

    total_time = time.time() - start_time
    state["invoice_data"] = all_results
    state["total_invoices_processed"] = len(all_results)
    state["average_time_per_invoice"] = round(total_time / max(1, len(all_results)), 2)
    state["node_details"] = None
    print("DEBUG: extract_all_node completed")
    return state

def extract_invoice_amount_node(state: dict) -> dict:
    import os, time
    print("DEBUG: Starting extract_invoice_amount_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json')]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}")

    invoice_amounts = []
    for json_file in json_files:
        json_data = robust_json_load(json_file)
        if not json_data:
            print(f"WARNING: Skipping {json_file} due to load error.")
            continue
        # Always extract from correct structure
        data = json_data.get("invoice_data", json_data)
        # Accept several possible fields for amount (case-insensitive)
        amount = None
        for section in data.get("non_table_text", {}).values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if "amount" in k.lower() or "total" in k.lower() or "payment" in k.lower():
                        try:
                            amount = float(str(v).replace(",", ""))
                            break
                        except Exception:
                            continue
            if amount is not None:
                break
        invoice_amounts.append(amount)

    total_time = time.time() - start_time
    state["invoice_amount"] = invoice_amounts
    state["total_invoices_processed"] = len(invoice_amounts)
    state["average_time_per_invoice"] = round(total_time / max(1, len(invoice_amounts)), 2)
    state["node_details"] = None
    print("DEBUG: extract_invoice_amount_node completed")
    return state

def extract_itemize_node(state: dict) -> dict:
    import os, time
    print("DEBUG: Starting extract_itemize_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json')]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}")

    all_items = []
    for json_file in json_files:
        json_data = robust_json_load(json_file)
        if not json_data:
            print(f"WARNING: Skipping {json_file} due to load error.")
            continue
        data = json_data.get("invoice_data", json_data)
        items_table = data.get("table", {}).get("items_table", {})
        all_items.append(items_table if items_table else {})
    total_time = time.time() - start_time
    state["items_table"] = all_items
    state["total_invoices_processed"] = len(all_items)
    state["average_time_per_invoice"] = round(total_time / max(1, len(all_items)), 2)
    state["node_details"] = None
    print("DEBUG: extract_itemize_node completed")
    return state

def result_node(state: dict) -> dict:
    # Always output dict with results for UI
    return {
        "final_response": {
            "processed_count": state.get("total_invoices_processed", 0),
            "average_processing_time_per_invoice": state.get("average_time_per_invoice", 0),
            "node_details": state.get("node_details", None),
            # Always include results
            "results": state.get("invoice_data", state.get("invoice_amount", state.get("items_table", [])))
        }
    }
