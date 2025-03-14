import os
import socket
import ssl
import webbrowser
import json
from flask import Flask, jsonify, request, send_from_directory
from threading import Timer
import service
import model_training.lite_predict as predition_model

from service import (
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

DEFAULT_DISTANCE_SET = {0.5, 1.0, 3.0, 5.0}
DEFAULT_DISTANCE = 3.0

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
        distance = float(request.args.get("distance", DEFAULT_DISTANCE))
        if distance not in DEFAULT_DISTANCE_SET:
            distance = DEFAULT_DISTANCE
    except (KeyError, ValueError):
        return jsonify({"error": "Missing or invalid lat/lng"}), 400

    result = recommend_parking(lat, lng, distance)
    return jsonify(result)


@app.route("/api/predict", methods=["GET"])
def api_predict():
    try:
        lot_id = int(request.args["lot_id"])
        predition_time = str(request.args["predition_time"])
        lopp_time = int(request.args["lopp_time"])

        print(f"lot_id : {lot_id}")
        print(f"predition_time : {predition_time}")
        print(f"lopp_time : {lopp_time}")

    except ValueError:
        return jsonify({"error": "Invalid hour_offset"}), 400

    result = service.predictparkinglot(predition_time, lot_id, lopp_time)
    return jsonify(result)


@app.route("/parking_lot", methods=["POST"])
def create_lot():
    """
    Create a new parking record：
    {
        "name": "Parking A",
        "address": "XX street",
        "latitude": 39.90,
        "longitude": 116.40,
        "total_spaces": 100,
        "available_spaces": 80
    }
    """
    data = request.json
    name = data.get("name")
    address = data.get("address")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    total_spaces = data.get("total_spaces")
    available_spaces = data.get("available_spaces")

    lot_id = service.create_parking_lot(
        name, address, latitude, longitude, total_spaces, available_spaces
    )
    return jsonify({"message": "Parking lot created", "lot_id": lot_id}), 201


@app.route("/parking_lot/<int:lot_id>", methods=["GET"])
def get_lot(lot_id):
    """get single record by lot_id"""
    row = service.get_parking_lot(lot_id)
    if row:
        # (lot_id, name, address, latitude, longitude, total_spaces, available_spaces, created_at, updated_at)
        return jsonify(
            {
                "lot_id": row[0],
                "name": row[1],
                "address": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "total_spaces": row[5],
                "available_spaces": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
        )
    else:
        return jsonify({"message": "Parking lot not found"}), 404


@app.route("/parking_lots", methods=["GET"])
def get_all_lots():
    """All parking lots"""
    rows = service.get_all_parking_lots()

    result = []
    for row in rows:
        result.append(
            {
                "lot_id": row[0],
                "name": row[1],
                "address": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "total_spaces": row[5],
                "available_spaces": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
        )
    return jsonify(result)


@app.route("/parking_lot/<int:lot_id>", methods=["PUT"])
def update_lot(lot_id):
    """
    Update a parking record by lot_id
    {
        "name": "Parking B",
        "address": "new street",
        "latitude": 39.91,
        "longitude": 116.41,
        "total_spaces": 200,
        "available_spaces": 150
    }
    """
    data = request.json
    name = data.get("name")
    address = data.get("address")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    total_spaces = data.get("total_spaces")
    available_spaces = data.get("available_spaces")

    # check if the record exists
    row = service.get_parking_lot(lot_id)
    if not row:
        return jsonify({"message": "Parking lot not found"}), 404

    service.update_parking_lot(
        lot_id, name, address, latitude, longitude, total_spaces, available_spaces
    )
    return jsonify({"message": "Parking lot updated"}), 200


@app.route("/parking_lot/<int:lot_id>", methods=["DELETE"])
def delete_lot(lot_id):
    """delete a parking record by lot_id"""
    row = service.get_parking_lot(lot_id)
    if not row:
        return jsonify({"message": "Parking lot not found"}), 404

    service.delete_parking_lot(lot_id)
    return jsonify({"message": "Parking lot deleted"}), 200


@app.route("/predictions", methods=["GET"])
def get_all_predictions():
    predictions = service.get_all_predictions()
    return jsonify(predictions)


@app.route("/prediction/<int:prediction_id>", methods=["GET"])
def get_prediction(prediction_id):
    prediction = service.get_prediction(prediction_id)
    if prediction:
        return jsonify(prediction)
    else:
        return jsonify({"message": "Prediction not found"}), 404


@app.route("/prediction", methods=["POST"])
def create_prediction():
    data = request.json
    prediction_id = service.create_prediction(
        data["lot_id"],
        data["prediction_time"],
        data.get("predicted_occupied_spaces"),
        data.get("predicted_available_spaces"),
        data.get("predicted_occupancy_rate"),
        data.get("confidence_score"),
        data.get("model_version"),
    )
    return (
        jsonify({"message": "Prediction created", "prediction_id": prediction_id}),
        201,
    )


@app.route("/prediction/<int:prediction_id>", methods=["PUT"])
def update_prediction(prediction_id):
    data = request.json
    service.update_prediction(
        prediction_id,
        data["lot_id"],
        data["prediction_time"],
        data.get("predicted_occupied_spaces"),
        data.get("predicted_available_spaces"),
        data.get("predicted_occupancy_rate"),
        data.get("confidence_score"),
        data.get("model_version"),
    )
    return jsonify({"message": "Prediction updated"})


@app.route("/prediction/<int:prediction_id>", methods=["DELETE"])
def delete_prediction(prediction_id):
    service.delete_prediction(prediction_id)
    return jsonify({"message": "Prediction deleted"})


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
    # predition_model
    predition_model.preload_all_model()

    Timer(1, open_browser).start()

    run_flask()

    input("Press Enter to exit...\n")
