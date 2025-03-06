import os
import socket
import ssl
import threading
import webbrowser
import math
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from threading import Timer

from service import (
    get_real_time_status,
    recommend_parking,
    predict_congestion,
)

# -------------------------------
# Configuration
# -------------------------------
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(BASE_DIR, "cert")
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "key.pem")

# Create a Flask application, using the current directory as a static file directory
app = Flask(__name__, static_folder=".", static_url_path="")

# -------------------------------
# Helper Functions
# -------------------------------


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return ip


# -------------------------------
# API Endpoints
# -------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# -------------------------------
# API
# -------------------------------
@app.route("/api/get-parking-data", methods=["GET"])
def get_parking_data():
    """
    Simulated backend interface
    From local JSON file
    """
    json_path = os.path.join(BASE_DIR, "json", "parking_data.json")
    try:
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"error": "Failed to read parking data", "message": str(e)}), 500
    return jsonify(data)


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    try:
        lat = float(request.args["lat"])
        lng = float(request.args["lng"])
    except (KeyError, ValueError):
        return jsonify({"error": "Missing or invalid lat/lng"}), 400

    result = recommend_parking(lat, lng)
    return jsonify(result)


@app.route("/api/predict", methods=["GET"])
def api_predict():
    """
    TODO Predicting congestion after X hours
    GET /api/predict?hour_offset=2
    """
    try:
        hour_offset = int(request.args.get("hour_offset", 0))
    except ValueError:
        return jsonify({"error": "Invalid hour_offset"}), 400

    result = predict_congestion(hour_offset)
    return jsonify(result)


# -------------------------------
# Start the server function
# -------------------------------
def run_server():
    # config SSL，use TLS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    print(f"Server running at https://{get_local_ip()}:{PORT}")
    # debug only development
    app.run(host="0.0.0.0", port=PORT, ssl_context=context, debug=True)


def open_browser():
    url = f"https://{get_local_ip()}:{PORT}"
    print(f"Opening {url} in default browser...")
    webbrowser.open(url)


# -------------------------------
# start Flask server
# -------------------------------
def run_flask():

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)

    # This must be called in the main thread app.run()
    print(f"Server running at https://{get_local_ip()}:{PORT}")
    # close debug/reloader
    app.run(
        host="0.0.0.0", port=PORT, ssl_context=context, debug=False, use_reloader=False
    )


# -------------------------------
# main
# -------------------------------
if __name__ == "__main__":
    Timer(1, open_browser).start()

    run_flask()

    input("Press Enter to exit...\n")
