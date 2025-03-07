# service.py

import sqlite3
import math
from datetime import datetime, timedelta

DB_PATH = "SmartParking.db"


def connect_db():
    """连Connect to the database and return the conn object. Context management is also available"""
    return sqlite3.connect(DB_PATH)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two latitude and longitude points (km)"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def get_real_time_status():
    """
    {
      "lot_id": ...,
      "name": ...,
      "total_spaces": ...,
      "available_spaces": ...,
      "congestion": ... (0~100%)
    }
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT lot_id, name, total_spaces, available_spaces FROM parkinglots"
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for lot_id, name, total, available in rows:
        if total > 0:
            congestion = ((total - available) / total) * 100
        else:
            congestion = 0
        results.append(
            {
                "lot_id": lot_id,
                "name": name,
                "total_spaces": total,
                "available_spaces": available,
                "congestion_percent": round(congestion, 1),
            }
        )
    return results


def recommend_parking(user_lat, user_lon):
    """
    Calculate based on user coordinates + parking lot congestion rate
    score=distance*congestion_factor,

    {
      "all_lots": [...],  # Score information for each lot
      "best_lot": {...}   # Best recommendation
    }
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT lot_id, name, latitude, longitude, total_spaces, available_spaces FROM parkinglots"
    )
    lots = cursor.fetchall()
    conn.close()

    all_lots = []
    best_score = None
    best_lot = None

    for lot_id, name, lat, lon, total, available in lots:
        distance = haversine(user_lat, user_lon, lat, lon)
        congestion_factor = (total - available) / total if total > 0 else 1
        score = distance * congestion_factor

        lot_info = {
            "lot_id": lot_id,
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "distance_km": round(distance, 2),
            "available_spaces": available,
            "total_spaces": total,
            "congestion_factor": round(congestion_factor, 2),
            "score": round(score, 2),
        }
        all_lots.append(lot_info)

        if best_score is None or score < best_score:
            best_score = score
            best_lot = lot_info

    return {"all_lots": all_lots, "best_lot": best_lot}


def predict_congestion(hour_offset):
    """
    Predict the parking lot occupancy after a specified hour (from the ai_predictions table)
    返回:
    {
      "time_range": "xxxx ~ xxxx",
      "predictions": [ {...}, {...} ]
    }
    """
    target_start = datetime.now() + timedelta(hours=hour_offset)
    target_end = target_start + timedelta(hours=1)
    start_str = target_start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = target_end.strftime("%Y-%m-%d %H:%M:%S")

    conn = connect_db()
    cursor = conn.cursor()
    query = """
    SELECT a.lot_id, p.name, a.prediction_time, a.predicted_occupied_spaces,
           a.predicted_available_spaces, a.predicted_occupancy_rate
    FROM ai_predictions a
    JOIN parkinglots p ON a.lot_id = p.lot_id
    WHERE a.prediction_time BETWEEN ? AND ?
    ORDER BY a.prediction_time
    """
    cursor.execute(query, (start_str, end_str))
    predictions = cursor.fetchall()
    conn.close()

    results = []
    for lot_id, lot_name, pred_time, occ, avail, occ_rate in predictions:
        results.append(
            {
                "lot_id": lot_id,
                "lot_name": lot_name,
                "prediction_time": pred_time,
                "predicted_occupied": occ,
                "predicted_available": avail,
                "occupancy_rate": round(occ_rate, 1),
                "status": "Full" if occ_rate >= 100 else "Not Full",
            }
        )

    return {"time_range": f"{start_str} ~ {end_str}", "predictions": results}
