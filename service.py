# service.py

import sqlite3
import math
from datetime import datetime, timedelta
import model_training.lite_predict as predition_model

DB_PATH = "SmartParking.db"

WEEK_END = "weekend"
WEEK_DAY = "weekday"

MODEL_LSTM = "lstm"
MODEL_DNN = "dnn"

FIFTEEN_MINUTES = 15 * 60


def connect_db():
    """Connect to the database and return the conn object. Context management is also available"""
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


def recommend_parking(user_lat, user_lon, filter_distance=3):
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
        if distance > filter_distance:
            continue
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
        if score > 0:
            if best_score is None or score < best_score:
                best_score = score
                best_lot = lot_info

    return {"all_lots": all_lots, "best_lot": best_lot}


def predict_congestion(hour_offset):
    """
    Predict the parking lot occupancy after a specified hour (from the ai_predictions table)

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


def create_parking_lot(
    name, address, latitude, longitude, total_spaces, available_spaces
):
    conn = connect_db()
    cursor = conn.cursor()

    insert_sql = """
    INSERT INTO parkinglots 
    (name, address, latitude, longitude, total_spaces, available_spaces) 
    VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.execute(
        insert_sql, (name, address, latitude, longitude, total_spaces, available_spaces)
    )

    conn.commit()
    conn.close()
    return cursor.lastrowid


def get_parking_lot(lot_id):
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    select_sql = "SELECT * FROM parkinglots WHERE lot_id = ?"
    cursor.execute(select_sql, (lot_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_parking_lots():
    conn = connect_db()
    cursor = conn.cursor()
    select_sql = "SELECT * FROM parkinglots"
    cursor.execute(select_sql)
    rows = cursor.fetchall()
    conn.close()
    return rows


def update_parking_lot(
    lot_id, name, address, latitude, longitude, total_spaces, available_spaces
):
    conn = connect_db()
    cursor = conn.cursor()
    update_sql = """
    UPDATE parkinglots
    SET name = ?,
        address = ?,
        latitude = ?,
        longitude = ?,
        total_spaces = ?,
        available_spaces = ?,
        updated_at = ?
    WHERE lot_id = ?
    """

    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        update_sql,
        (
            name,
            address,
            latitude,
            longitude,
            total_spaces,
            available_spaces,
            updated_at,
            lot_id,
        ),
    )
    conn.commit()
    conn.close()


def delete_parking_lot(lot_id):
    conn = connect_db()
    cursor = conn.cursor()
    delete_sql = "DELETE FROM parkinglots WHERE lot_id = ?"
    cursor.execute(delete_sql, (lot_id,))
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ai_predictions")
    rows = cursor.fetchall()
    conn.close()
    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in rows
    ]


def get_prediction(prediction_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM ai_predictions WHERE prediction_id = ?", (prediction_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(zip([column[0] for column in cursor.description], row))
    return None


def create_prediction(
    lot_id,
    prediction_time,
    predicted_occupied_spaces,
    predicted_available_spaces,
    predicted_occupancy_rate,
    confidence_score,
    model_version,
):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ai_predictions (lot_id, prediction_time, predicted_occupied_spaces, predicted_available_spaces, 
                                   predicted_occupancy_rate, confidence_score, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            lot_id,
            prediction_time,
            predicted_occupied_spaces,
            predicted_available_spaces,
            predicted_occupancy_rate,
            confidence_score,
            model_version,
        ),
    )
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id


def update_prediction(
    prediction_id,
    lot_id,
    prediction_time,
    predicted_occupied_spaces,
    predicted_available_spaces,
    predicted_occupancy_rate,
    confidence_score,
    model_version,
):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE ai_predictions
        SET lot_id = ?, prediction_time = ?, predicted_occupied_spaces = ?, predicted_available_spaces = ?, 
            predicted_occupancy_rate = ?, confidence_score = ?, model_version = ?
        WHERE prediction_id = ?
    """,
        (
            lot_id,
            prediction_time,
            predicted_occupied_spaces,
            predicted_available_spaces,
            predicted_occupancy_rate,
            confidence_score,
            model_version,
            prediction_id,
        ),
    )
    conn.commit()
    conn.close()


def delete_prediction(prediction_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM ai_predictions WHERE prediction_id = ?", (prediction_id,)
    )
    conn.commit()
    conn.close()


def predictparkinglot(prediction_time, parkinglot_id, lopp_time, score):
    try:
        # Parse the string as a datetime object, and then take out the date part
        format_prediction_time = datetime.strptime(prediction_time, "%Y-%m-%d %H:%M:%S")

    except ValueError:
        return "The date format is wrong, please use the correct format, eg：YYYY-MM-DD HH:MM:SS"

    inputdatalist = get_parkingtransactions_by_time(
        format_prediction_time, parkinglot_id
    )

    # Default model type
    model_type = MODEL_LSTM
    model = get_model(format_prediction_time, model_type)
    print("inputdatalist", inputdatalist)
    print("model", model)

    predict = predition_model.predict_wrapper(
        model, inputdatalist, lopp_time, parkinglot_id
    )
    print("predict", predict)

    # Save the prediction result to the database

    parking_lot = get_parking_lot(parkinglot_id)

    print("parking_lot", parking_lot)
    create_prediction(
        parkinglot_id,
        prediction_time,
        predict,
        parking_lot["available_spaces"],
        parking_lot["available_spaces"] / parking_lot["total_spaces"],
        score,
        model_type,
    )
    return predict


def check_day(date_obj):
    # weekday() Method: Return 0 on Monday ,..., 6 on Sunday
    if date_obj.date().weekday() >= 5:
        return WEEK_END
    else:
        return WEEK_DAY


def get_model(prediction_time, model_type=MODEL_LSTM):
    day_type = check_day(prediction_time)
    """
    modeltype:
            g_model_type_lstm_weekdays
            g_model_type_lstm_weekends
            g_model_type_dnn_weekdays
            g_model_type_dnn_weekends

    """
    model_mapping = {
        WEEK_END: {
            MODEL_LSTM: predition_model.g_model_type_lstm_weekends,
            MODEL_DNN: predition_model.g_model_type_dnn_weekends,
        },
        WEEK_DAY: {
            MODEL_LSTM: predition_model.g_model_type_lstm_weekdays,
            MODEL_DNN: predition_model.g_model_type_dnn_weekdays,
        },
    }

    try:
        model = model_mapping[day_type][model_type]
    except KeyError:
        raise ValueError(
            f"Invalid date type or model type: day_type={day_type}, model_type={model_type}"
        )

    return model


def get_parkingtransactions_by_time(end_datatime, parking_lot_id):
    start_time = end_datatime - timedelta(hours=16)

    print(f"start_time({start_time}) - end_datatime({end_datatime})")

    conn = connect_db()
    cursor = conn.cursor()

    query = """
        WITH RECURSIVE intervals AS ( 
        SELECT CAST(strftime('%s', ?) / 900 AS INT) AS interval_15 
        UNION ALL
        SELECT interval_15 + 1
        FROM intervals
        WHERE interval_15 < (CAST(strftime('%s', ?) / 900 AS INT)) - 1
        )
        SELECT 
        intervals.interval_15,
        COALESCE(pt.total_count, 0) AS total_count
        FROM intervals
        LEFT JOIN (
        SELECT 
            CAST(strftime('%s', entry_time) / 900 AS INT) AS interval_15,
            COUNT(*) AS total_count
        FROM parkingtransactions
        WHERE parking_lot_id = ? 
            AND datetime(entry_time) >= ? 
            AND datetime(entry_time) <=? 
        GROUP BY interval_15
        ) pt ON intervals.interval_15 = pt.interval_15
        ORDER BY intervals.interval_15;
    """
    params = (start_time, end_datatime, parking_lot_id, start_time, end_datatime)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    print("rows", rows)
    value_list = [row[1] for row in rows]
    return value_list
