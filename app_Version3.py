import os
import json
import logging
from flask import Flask, request, render_template, jsonify
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

flask_app = Flask(__name__)
databricks = WorkspaceClient()

@flask_app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@flask_app.route('/invoice_run', methods=['POST'])
def invoice_run():
    try:
        req_data = request.get_json()
        input_path = req_data.get('input_path')
        extract_all = req_data.get('extract_all', False)
        extract_invoice_amount = req_data.get('extract_invoice_amount', False)
        extract_itemise = req_data.get('extract_itemise', False)

        if not input_path:
            return jsonify({"status": "Error", "error": "Input path is required"}), 400

        if not input_path.startswith("/dbfs/"):
            input_path = f"/dbfs/FileStore/invoicesense/{input_path}"

        run_id = databricks.jobs.run_now(
            job_id=899879803593620,
            notebook_params={
                "input_path": input_path,
                "extract_all": str(extract_all).lower(),
                "extract_invoice_amount": str(extract_invoice_amount).lower(),
                "extract_itemise": str(extract_itemise).lower()
            }
        ).run_id

        return jsonify({"run_id": run_id})

    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

@flask_app.route('/check_invoice_status', methods=['POST'])
def check_invoice_status():
    data = request.get_json()
    run_id = data.get('run_id')
    try:
        run_status = databricks.jobs.get_run(run_id)
        if run_status.state.life_cycle_state == RunLifeCycleState.TERMINATED:
            if run_status.state.result_state == RunResultState.SUCCESS:
                notebook_output = databricks.jobs.get_run_output(run_status.tasks[0].run_id).notebook_output
                if notebook_output.result:
                    try:
                        result = json.loads(notebook_output.result)
                        return jsonify({
                            "status": "Succeed",
                            "result": result
                        })
                    except json.JSONDecodeError:
                        return jsonify({
                            "status": "Succeed",
                            "result": {"raw_output": notebook_output.result}
                        })
            else:
                return jsonify({
                    "status": "Not-Succeed",
                    "result": f"Job failed: {run_status.state.state_message}"
                })
        else:
            return jsonify({
                "status": "Not-Finish",
                "result": f"Job status: {run_status.state.life_cycle_state.value}"
            })
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)})

@flask_app.route('/list_images', methods=['GET'])
def list_images():
    try:
        files = list(databricks.dbfs.list("/dbfs/FileStore/invoicesense"))
        image_files = [f.path.split('/')[-1] for f in files if f.path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg'))]
        folders = [f.path.split('/')[-1] for f in files if f.is_dir]
        return jsonify({"status": "success", "images": image_files, "folders": folders}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=False)