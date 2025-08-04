def extract_all_node(state: dict) -> dict:
    import time
    import os
    import json

    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json') and fname != "processing_stats.json"
    ]
    tabular_count = invoice_count = itemise_count = 0
    all_data = []

    for json_file in json_files:
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                if not isinstance(json_data, dict):
                    continue
            except Exception:
                continue

            result = analyzer.extract_all(json_data)
            all_data.append(result)
            if result.get("items_table"):
                tabular_count += 1
                itemise_count += 1
            if analyzer.extract_invoice_amount(json_data) is not None:
                invoice_count += 1

    total_time = time.time() - start_time
    total_count = len(all_data)
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
    import os
    import json

    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json') and fname != "processing_stats.json"
    ]

    invoice_amounts = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                if not isinstance(json_data, dict):
                    continue
            except Exception:
                continue

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
    import os
    import json

    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    process_images(image_folder, output_folder, batch_size, num_images)
    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json') and fname != "processing_stats.json"
    ]

    itemized_data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                if not isinstance(json_data, dict):
                    continue
            except Exception:
                continue

            itemized_data.append(analyzer.extract_itemize(json_data))

    total_time = time.time() - start_time
    count = len(itemized_data)
    state["items_table"] = itemized_data
    state["total_invoices_processed"] = count
    state["average_time_per_invoice"] = round(total_time / count, 2) if count else 0
    return state
