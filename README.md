# 6. Workflow Invocation and Result Formatting

initial_state = {
    "input_path": input_path,
    "extract_all": extract_all,
    "extract_invoice_amount": extract_invoice_amount,
    "extract_itemise": extract_itemise
}
final_state = invoice_workflow.invoke(initial_state)

result_data = {
    "processed_count": final_state['final_response']['processed_count'],
    "average_processing_time_per_invoice": final_state['final_response']['average_processing_time_per_invoice'],
    "node_details": final_state['final_response']['node_details']
}

try:
    dbutils.notebook.exit(json.dumps({"status": "success", "result": result_data}))
except Exception:
    print(json.dumps({"status": "success", "result": result_data}))
