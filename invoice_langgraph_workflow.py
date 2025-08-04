# Databricks notebook source
# DBTITLE 1,Importing required libraries
# MAGIC %pip install torch torchvision transformers datasets opencv-python faiss-cpu av==14.4.0 numpy==1.26.4 pillow qwen_vl_utils langgraph

from PIL import Image
import requests
import torch
import os
import json
import time
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2VLConfig
from typing import TypedDict, Optional, Any, List, Dict

# COMMAND ----------

# DBTITLE 1,Prompt (in separate cell or file if desired)
invoice_prompt = '''You are an image analysis expert and an invoice image is provided to you. Your task is to extract all information from the image and organize it into a structured JSON format.

The extracted data should be categorized into:

1. Tabular Data : A table is defined as a grid-like arrangement of data organized into rows and columns, often with headers. Ignore any surrounding text, logos, signatures, or decorative elements when extracting tabular data. It typically includes line items such as Quantity, Description, Unit Price, Discount and Amount.

2. Non-Tabular Text : Non-tabular text refers to any content that is not organized in a grid-like structure of rows and columns. This includes paragraphs, headers, footers, labels, annotations, and any standalone text blocks. Ignore tables, charts, and graphical elements. This tipically includes metadata such as Invoice Number, Invoice Date, PO Number, Due Date, Billing Addresses and Shipping Addresses, Company Information, Tax Rate, Calculated Tax Amounts, Discount rate, Total Discount Value, Subtotal, Total, and Payment Terms.

3. Non_table_image Elements : Any other elements such as Signatures, lLgos, or Stamps, described with associated text. Simply ignore this and exclude from JSON.

Important Instructions:
Use your understanding of invoice structure to group non-tabular data into meaningful sections such as:
  -"Invoice Details" – e.g. invoice number, date, PO number, due date
  -"Billing Info" – e.g. bill to and ship to names and addresses
  -"Organization Info"
  -"Financial Summary" – e.g., subtotal, tax rate, tax amount, GST rate, GST Amount,  total amount due
  -"Terms and Conditions" – e.g., payment terms or due in days

# If any rate%, calculated tax, subtotal, or total is found, include it under "Financial Summary" as key-value pairs.

# Insome cases the field names and corresponding values are provided side by side, in other cases the field names are not provided and the values are scattered across the image. In such cases, use your understanding of invoice structure to group the values into meaningful sections and make sure no information is missed.

# In some cases the field names are not provided rather a total is given below some table columms so make sure this this information also comes under Finacial Summary as key-value pairs like Total Payment Amount, Total Discount etc .  

# The company name and address are found under 'From'.

# Do not try to read the company logo or any other image, skip if not found.

# You can omit a section from sample JSON below if it is not present in the image.

# Make it double sure that no data is left in the image without being added to the JSON.

Ensure the output is valid JSON and follows this structure:    

{
  "invoice_data": {
    "table": {
      "items_table": {
        "headers": ["Key1", "Key2", "Key3", "Key4"],
        "rows": [
            {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3",
                "key4": "value4"
            },
            {...}]
      }
    },
    "non_table_text": {
      "section_name": { "key": "...", "data": { ... } },
      "section_2": { "group_name": "...", "data": { ... } },
      ...
    }               
  }
}
'''

# COMMAND ----------

# DBTITLE 1,State Schema for Workflow Pipeline Agent
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
    input_path = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data"
    extract_all = True
    extract_invoice_amount = False
    extract_itemise = False

# COMMAND ----------

# DBTITLE 1,User Input Node
def user_input_node(state: InvoiceState) -> InvoiceState:
    input_path = dbutils.widgets.get("input_path")
    if not input_path:
        raise ValueError("Input path is required.")

    extract_all = dbutils.widgets.get("extract_all") == "true"
    extract_invoice_amount = dbutils.widgets.get("extract_invoice_amount") == "true"
    extract_itemise = dbutils.widgets.get("extract_itemise") == "true"

    state["input_path"] = input_path
    state["extract_all"] = extract_all
    state["extract_invoice_amount"] = extract_invoice_amount
    state["extract_itemise"] = extract_itemise
    return state

# COMMAND ----------

# DBTITLE 1,Model Initialization (Qwen2-VL)
model = None
processor = None

def initialize_model():
    global model, processor
    if model is None or processor is None:
        print("Initializing model...")
        local_model_path = "/dbfs/FileStore/model/Qwen/Qwen2-VL-7B-Instruct"
        config = Qwen2VLConfig.from_pretrained(local_model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        print("Model and processor loaded.")
    return model, processor

model, processor = initialize_model()

# COMMAND ----------

# DBTITLE 1,InvoiceAnalyzer class
class InvoiceAnalyzer:
    def __init__(self, input_path: str, prompt: str):
        self.input_path = input_path
        self.prompt_template = prompt

    def extract_all(self, json_data: Dict) -> Dict:
        result = {
            "items_table": json_data.get("invoice_data", {}).get("table", {}).get("items_table", {}),
            "non_table_text": json_data.get("invoice_data", {}).get("non_table_text", {}),
            "non_table_image": json_data.get("invoice_data", {}).get("non_table_image", {})
        }
        return result

    def extract_itemize(self, json_data: Dict) -> Dict:
        result = {
            "items_table": json_data.get("invoice_data", {}).get("table", {}).get("items_table", {})
        }
        return result

    def extract_invoice_amount(self, json_data: Dict) -> Optional[float]:
        non_table_text_data = json_data.get("invoice_data", {}).get("non_table_text", {})
        amount = None
        for section in non_table_text_data.values():
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
        return amount

# COMMAND ----------

# DBTITLE 1,Robust JSON Loader
def robust_json_load(json_file):
    """Always returns a list of dicts, never skips a file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Handle double-encoded JSON (string inside JSON)
            tries = 0
            while isinstance(data, str) and tries < 3:
                data = json.loads(data)
                tries += 1
    except Exception as e:
        return [{"raw_load_error": str(e), "file": json_file}]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d if isinstance(d, dict) else {"raw_entry": d, "file": json_file} for d in data]
    return [{"raw_entry": data, "file": json_file}]

# COMMAND ----------

# DBTITLE 1,Image and Batch Processor
def process_image_batch(image_paths):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Resize the image
            new_size = (int(img.width * 0.5), int(img.height * 0.5))
            resized_image = img.resize(new_size, Image.LANCZOS)
            images.append(resized_image.convert("RGB"))
    analyzer = InvoiceAnalyzer("", invoice_prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": analyzer.prompt_template}
            ],
        }
        for image in images
    ]
    texts = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.cuda(non_blocking=True) if hasattr(v, "cuda") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    results = []
    for image_path, description in zip(image_paths, output_texts):
        try:
            json_data = json.loads(description)
        except Exception:
            json_data = {}
        results.append((os.path.splitext(os.path.basename(image_path))[0], json_data))
    return results

def save_json(output_folder, base_name, json_data):
    output_path = os.path.join(output_folder, f"{base_name}.json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def process_images(image_folder, output_folder, batch_size=2, num_images=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if num_images is not None:
        image_paths = image_paths[:num_images]
    total_images = len(image_paths)
    total_time = 0
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        start_time = time.time()
        batch_results = process_image_batch(batch_paths)
        batch_time = time.time() - start_time
        total_time += batch_time
        for base_name, json_data in batch_results:
            save_json(output_folder, base_name, json_data)
    avg_time_per_image = total_time / total_images if total_images else 0
    print(f"Average processing time per image: {avg_time_per_image:.2f} seconds")

# COMMAND ----------

# DBTITLE 1,Extract Nodes
def extract_all_node(state: dict) -> dict:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"], invoice_prompt)
    output_folder = state.get("output_folder") or "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10
    process_images(state["input_path"], output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json')]
    invoice_data = []
    for json_file in json_files:
        for invoice in robust_json_load(json_file):
            invoice_data.append(invoice)
    total_time = time.time() - start_time
    state["invoice_data"] = invoice_data
    state["total_invoices_processed"] = len(invoice_data)
    state["average_time_per_invoice"] = round(total_time / max(1, len(invoice_data)), 2)
    return state

def extract_invoice_amount_node(state: dict) -> dict:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"], invoice_prompt)
    output_folder = state.get("output_folder") or "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10
    process_images(state["input_path"], output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json')]
    invoice_amounts = []
    for json_file in json_files:
        for invoice in robust_json_load(json_file):
            data = invoice.get("invoice_data", invoice)
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
    return state

def extract_itemize_node(state: dict) -> dict:
    start_time = time.time()
    analyzer = InvoiceAnalyzer(state["input_path"], invoice_prompt)
    output_folder = state.get("output_folder") or "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5
    num_images = 10
    process_images(state["input_path"], output_folder, batch_size, num_images)
    json_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.json')]
    items_tables = []
    for json_file in json_files:
        for invoice in robust_json_load(json_file):
            data = invoice.get("invoice_data", invoice)
            items_table = data.get("table", {}).get("items_table", {})
            items_tables.append(items_table if items_table else {})
    total_time = time.time() - start_time
    state["items_table"] = items_tables
    state["total_invoices_processed"] = len(items_tables)
    state["average_time_per_invoice"] = round(total_time / max(1, len(items_tables)), 2)
    return state

# COMMAND ----------

# DBTITLE 1,Result Node
def result_node(state: InvoiceState) -> InvoiceState:
    state['final_response'] = {
        "processed_count": state.get("total_invoices_processed", 0),
        "average_processing_time_per_invoice": state.get("average_time_per_invoice", 0),
    }
    return state

# COMMAND ----------

# DBTITLE 1,LangGraph Workflow Setup
from langgraph.graph import StateGraph, END

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

# COMMAND ----------

# DBTITLE 1,Run the Workflow and Display Only the Timing
initial_state = {
    "input_path": input_path,
    "extract_all": extract_all,
    "extract_invoice_amount": extract_invoice_amount,
    "extract_itemise": extract_itemise
}
final_state = invoice_workflow.invoke(initial_state)

# Defensive: always expect dict, never call `.get` on a string.
final_result = final_state['final_response'] if isinstance(final_state, dict) and isinstance(final_state.get('final_response', None), dict) else {
    "processed_count": 0,
    "average_processing_time_per_invoice": 0,
    "node_details": None
}

result_data = {
    "processed_count": final_result['processed_count'],
    "average_processing_time_per_invoice": final_result['average_processing_time_per_invoice'],
}

try:
    dbutils.notebook.exit(json.dumps({"status": "success", "result": result_data}))
except Exception:
    print(json.dumps({"status": "success", "result": result_data}))