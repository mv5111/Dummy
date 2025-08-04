def extract_all_node(state: dict) -> dict:
    import time
    import os
    import json

    print("DEBUG: Starting extract_all_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    print(f"DEBUG: image_folder={image_folder}")
    print(f"DEBUG: output_folder={output_folder}")

    process_images(image_folder, output_folder, batch_size, num_images)
    print("DEBUG: Finished process_images")

    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json')
    ]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}:")
    for jf in json_files:
        print("  -", jf)

    tabular_count = invoice_count = itemise_count = 0
    all_data = []

    for json_file in json_files:
        print(f"DEBUG: Processing file {json_file}")
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                print(f"DEBUG: Loaded JSON from {json_file}")
                if not isinstance(json_data, dict):
                    print(f"WARNING: JSON in {json_file} is not a dict, skipping.")
                    continue
            except Exception as e:
                print(f"ERROR: Failed to load JSON from {json_file}: {e}")
                continue

            result = analyzer.extract_all(json_data)
            print(f"DEBUG: extract_all result for {json_file}: {result}")
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
    print(f"DEBUG: Node summary: invoice_data_len={len(all_data)}, tabular_count={tabular_count}, invoice_count={invoice_count}, itemise_count={itemise_count}")
    state["invoice_data"] = all_data
    state["total_invoices_processed"] = total_count
    state["average_time_per_invoice"] = round(total_time / total_count, 2) if total_count else 0
    state["node_details"] = node_details
    print("DEBUG: extract_all_node completed")
    return state

def extract_invoice_amount_node(state: dict) -> dict:
    import time
    import os
    import json

    print("DEBUG: Starting extract_invoice_amount_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    print(f"DEBUG: image_folder={image_folder}")
    print(f"DEBUG: output_folder={output_folder}")

    process_images(image_folder, output_folder, batch_size, num_images)
    print("DEBUG: Finished process_images")

    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json')
    ]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}:")
    for jf in json_files:
        print("  -", jf)

    invoice_amounts = []
    for json_file in json_files:
        print(f"DEBUG: Processing file {json_file}")
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                print(f"DEBUG: Loaded JSON from {json_file}")
                if not isinstance(json_data, dict):
                    print(f"WARNING: JSON in {json_file} is not a dict, skipping.")
                    continue
            except Exception as e:
                print(f"ERROR: Failed to load JSON from {json_file}: {e}")
                continue

            amount = analyzer.extract_invoice_amount(json_data)
            print(f"DEBUG: extract_invoice_amount for {json_file}: {amount}")
            if amount is not None:
                invoice_amounts.append(amount)

    total_time = time.time() - start_time
    count = len(invoice_amounts)
    print(f"DEBUG: Node summary: invoice_amounts={invoice_amounts}, count={count}")
    state["invoice_amount"] = invoice_amounts
    state["total_invoices_processed"] = count
    state["average_time_per_invoice"] = round(total_time / count, 2) if count else 0
    print("DEBUG: extract_invoice_amount_node completed")
    return state

def extract_itemize_node(state: dict) -> dict:
    import time
    import os
    import json

    print("DEBUG: Starting extract_itemize_node")
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"])
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10

    print(f"DEBUG: image_folder={image_folder}")
    print(f"DEBUG: output_folder={output_folder}")

    process_images(image_folder, output_folder, batch_size, num_images)
    print("DEBUG: Finished process_images")

    json_files = [
        os.path.join(output_folder, fname)
        for fname in os.listdir(output_folder)
        if fname.endswith('.json')
    ]
    print(f"DEBUG: Found {len(json_files)} JSON files in {output_folder}:")
    for jf in json_files:
        print("  -", jf)

    itemized_data = []
    for json_file in json_files:
        print(f"DEBUG: Processing file {json_file}")
        with open(json_file, 'r') as file:
            try:
                json_data = json.load(file)
                print(f"DEBUG: Loaded JSON from {json_file}")
                if not isinstance(json_data, dict):
                    print(f"WARNING: JSON in {json_file} is not a dict, skipping.")
                    continue
            except Exception as e:
                print(f"ERROR: Failed to load JSON from {json_file}: {e}")
                continue

            itemized = analyzer.extract_itemize(json_data)
            print(f"DEBUG: extract_itemize for {json_file}: {itemized}")
            itemized_data.append(itemized)

    total_time = time.time() - start_time
    count = len(itemized_data)
    print(f"DEBUG: Node summary: itemized_data_length={len(itemized_data)}, count={count}")
    state["items_table"] = itemized_data
    state["total_invoices_processed"] = count
    state["average_time_per_invoice"] = round(total_time / count, 2) if count else 0
    print("DEBUG: extract_itemize_node completed")
    return state
