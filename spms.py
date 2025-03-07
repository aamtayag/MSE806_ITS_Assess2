

import sqlite3
import math
from datetime import datetime, timedelta

# ------------------------------
# Helper Functions
# ------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points (in km)
    on the earth (specified in decimal degrees).
    """
    # convert decimal degrees to radians
    R = 6371  # Radius of earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_real_time_status(conn):
    """
    Retrieves real-time parking congestion status for each parking lot.
    Congestion is computed as: (total_spaces - available_spaces) / total_spaces * 100%
    """
    cursor = conn.cursor()
    cursor.execute("SELECT lot_id, name, total_spaces, available_spaces FROM parkinglots")
    rows = cursor.fetchall()
    print("\nReal-Time Parking Status:")
    for lot_id, name, total, available in rows:
        if total > 0:
            congestion = ((total - available) / total) * 100
        else:
            congestion = 0
        print(f"Lot ID: {lot_id}, Name: {name}")
        print(f"   Total Spaces: {total}, Available: {available}, Congestion: {congestion:.1f}%")
    print()

def recommend_parking(conn, user_lat, user_lon):
    """
    Recommends the best parking lot based on the user's location and congestion.
    The recommendation score is computed as: score = distance (km) * congestion_factor,
    where congestion_factor = (total_spaces - available_spaces) / total_spaces.
    Lower scores are better.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT lot_id, name, latitude, longitude, total_spaces, available_spaces FROM parkinglots")
    lots = cursor.fetchall()
    
    best_score = None
    best_lot = None
    
    print("\nParking Recommendations:")
    for lot in lots:
        lot_id, name, lat, lon, total, available = lot
        distance = haversine(user_lat, user_lon, lat, lon)
        congestion_factor = (total - available) / total if total > 0 else 1         # original
        # congestion_factor = (total - available) / total if total > 0 else 0         # modified this from 1 to 0
        score = distance * congestion_factor
        print(f"Lot {name} (ID: {lot_id}) -> Distance: {distance:.2f} km, Congestion Factor: {congestion_factor:.2f}, Score: {score:.2f}")
        # if score > 0:                                                               # added this IF statement
        if best_score is None or score < best_score:
            best_score = score
            best_lot = lot
    
    if best_lot:
        lot_id, name, lat, lon, total, available = best_lot
        print(f"\nRecommended Parking Lot: {name} (ID: {lot_id})")
        print(f"Distance: {haversine(user_lat, user_lon, lat, lon):.2f} km, Available Spaces: {available}/{total}\n")
    else:
        print("No parking lots available.\n")

def predict_congestion(conn, hour_offset):
    """
    Provides prediction of parking congestion for each parking lot for a specified hour offset
    from the current time. It queries the ai_predictions table for predictions where the
    prediction_time is within the target hour.
    """
    target_start = datetime.now() + timedelta(hours=hour_offset)
    target_end = target_start + timedelta(hours=1)
    
    cursor = conn.cursor()
    # Convert datetime objects to strings (SQLite default datetime format)
    start_str = target_start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = target_end.strftime("%Y-%m-%d %H:%M:%S")
    
    query = """
    SELECT a.lot_id, p.name, a.prediction_time, a.predicted_occupied_spaces, a.predicted_available_spaces, a.predicted_occupancy_rate 
    FROM ai_predictions a 
    JOIN parkinglots p ON a.lot_id = p.lot_id
    WHERE a.prediction_time BETWEEN ? AND ?
    ORDER BY a.prediction_time
    """
    cursor.execute(query, (start_str, end_str))
    predictions = cursor.fetchall()
    
    print(f"\nParking Congestion Predictions for the period {start_str} to {end_str}:")
    if predictions:
        for pred in predictions:
            lot_id, lot_name, pred_time, occ, avail, occ_rate = pred
            status = "Full" if occ_rate >= 100 else "Not Full"
            print(f"Lot: {lot_name} (ID: {lot_id}) at {pred_time}")
            print(f"   Predicted Occupied: {occ}, Available: {avail}, Occupancy Rate: {occ_rate:.1f}% -> {status}")
    else:
        print("No predictions available for this period. Consider checking back later or adjusting the query.")
    print()

# ------------------------------
# Main Application
# ------------------------------

def main():
    # Connect to the SQLite database (make sure SmartParking.db exists with the proper schema)
    conn = sqlite3.connect("SmartParking.db")
    
    while True:
        print("Smart Parking Management System")
        print("1. View Real-Time Parking Status")
        print("2. Get Parking Recommendation")
        print("3. Predict Parking Congestion")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            get_real_time_status(conn)
        elif choice == "2":
            try:
                user_lat = float(input("Enter your current latitude: ").strip())
                user_lon = float(input("Enter your current longitude: ").strip())
                recommend_parking(conn, user_lat, user_lon)
            except ValueError:
                print("Invalid coordinates. Please try again.\n")
        elif choice == "3":
            try:
                hour_offset = int(input("Enter number of hours from now for the prediction (e.g., 1, 2, 3): ").strip())
                predict_congestion(conn, hour_offset)
            except ValueError:
                print("Invalid input. Please enter a valid number of hours.\n")
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid option. Please choose between 1 and 4.\n")
    
    conn.close()

if __name__ == "__main__":
    main()

