import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

STATE_FILE = "/root/agent_state.json"

@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.json
    user_input = data.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "输入为空"}), 400

    try:
        env = os.environ.copy()
        env["HABITAT_SIM_LOG"] = "quiet"
        env["MAGNUM_LOG"] = "quiet"
        env["EGL_DEVICE_ID"] = "0"

        result = subprocess.run(
            ["python", "/root/navigator_process.py", user_input],
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )

        print("returncode:", result.returncode)

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "message": "导航进程崩溃，请重试",
                "frames": []
            })

        stdout = result.stdout.strip()
        json_start = stdout.rfind('{')
        if json_start == -1:
            return jsonify({
                "success": False,
                "message": "导航进程异常，请重试",
                "frames": []
            })

        json_str = stdout[json_start:]
        output = json.loads(json_str)
        return jsonify(output)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    return jsonify({"status": "reset done"})

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"running": False, "done": True, "message": "", "frame_count": 0})

@app.route("/frames", methods=["GET"])
def get_frames():
    return jsonify({"frames": [], "total": 0})

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    from pyngrok import ngrok
    ngrok.set_auth_token("3DLJNbZI5bZhxqZv3P9J5X1aDXu_2XbQRSzLDKFruUeyJjSz")
    public_url = ngrok.connect(6006)
    print("=" * 50)
    print(f"公网访问地址：{public_url}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=6006, debug=False)