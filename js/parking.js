// parking.js
// Global array to store all parking lot data loaded from JSON or an API
let parkingData = [];
let parkingMarkers = [];


/**
 * Calculates the distance (in km) between two latitude-longitude points
 * on Earth using the Haversine formula.
 * @param {number} lat1 - Latitude of the first point
 * @param {number} lon1 - Longitude of the first point
 * @param {number} lat2 - Latitude of the second point
 * @param {number} lon2 - Longitude of the second point
 * @returns {number} Distance in kilometers
 */
function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in kilometers
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// Clean old markers before loading new ones
function cleanMarkersAndData() {
    parkingMarkers.forEach(marker => marker.setMap(null));
    parkingMarkers = [];
    parkingData = [];
}

// Get the value of the distance selection box
function getSelectedDistance() {
    const distanceSelect = document.getElementById("distance-select");
    return distanceSelect.value;
}


/**
 * Loads parking data from either a local JSON (if DEBUG_MODE is true)
 * or from an API endpoint (if DEBUG_MODE is false).
 *
 * This function depends on:
 *   - currentLocation: the global object in map.js with { lat, lng, address }
 *   - map (the Google Map object)
 *   - DEBUG_MODE (to decide data source)
 */

function loadParkingData() {
    console.log("Current location in parking.js:", currentLocation);

    cleanMarkersAndData();

    fetch(`${window.API_CONFIG.API_BASE_URL}/api/recommend?lat=${currentLocation.lat}&lng=${currentLocation.lng}&distance=${getSelectedDistance()}`)
        .then(response => response.json())
        .then(data => {
            // Compute the distance from the user's currentLocation to each parking lot
            const allLots = data.all_lots;

            const bestLots = data.best_lot;
            console.log("Best Lots:", bestLots);

            if (!Array.isArray(allLots)) {
                throw new Error("No 'all_lots' array found in response");
            }

            parkingData = allLots.map(lot => ({
                ...lot,
                distance: getDistance(
                    currentLocation.lat,
                    currentLocation.lng,
                    lot.latitude,
                    lot.longitude
                )
            }));

            // Sort parking lots from closest to farthest
            parkingData.sort((a, b) => {
                if (b.score !== a.score) {
                    return a.score - b.score; // score from low to high
                } else {
                    return a.distance - b.distance; // distance from low to high
                }
            });

            // Create a Marker and InfoWindow for each parking lot
            parkingData.forEach(lot => {
                const marker = new google.maps.Marker({
                    position: { lat: lot.latitude, lng: lot.longitude },
                    map: map, // 'map' should be global from map.js
                    title: lot.name
                });

                marker.addListener("click", () => {
                    if (currentInfoWindow) {
                        currentInfoWindow.close();
                    }
                    setInfoWindow(lot);
                    currentInfoWindow.open(map, marker);
                });
                parkingMarkers.push(marker);
                lot.marker = marker;
                // 

                if (bestLots.lot_id === lot.lot_id) {
                    lot.best_lot = true;
                }

            });

            // Update the list in the side panel
            updateParkingList();
        })
        .catch(error => {
            console.error("Failed to load parking data:", error);
        });
}

function setInfoWindow(lot) {
    currentInfoWindow.setContent(`
        <h3>${lot.name}</h3>
        <p>Available: ${lot.available_spaces} / ${lot.total_spaces}</p>
        <p>Distance: ${lot.distance.toFixed(2)} km</p>
        <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>
    `);
}


/**
 * Updates the DOM list (e.g., <ul id="parking-items">) with the current sorted parkingData.
 */
function updateParkingList() {
    const parkingList = document.getElementById('parking-items');
    if (!parkingList) {
        console.warn("Element with id='parking-items' not found in the DOM.");
        return;
    }
    parkingList.innerHTML = "";

    // Empty state
    if (parkingData.length === 0) {
        const noDataItem = document.createElement('li');
        noDataItem.classList.add('parking-item');
        noDataItem.textContent = "No parking lots found nearby.";
        parkingList.appendChild(noDataItem);
        return;
    }

    parkingData.forEach((lot, index) => {
        const item = document.createElement('li');
        item.classList.add('parking-item');

        if (lot.best_lot) {
            item.classList.add('best-lot');
        }

        item.innerHTML = `
      <strong>${lot.name} ${lot.best_lot ? "<span class='best-label'>🏅 Best 🏅</span>" : ""}</strong><br>
      Distance: ${lot.distance.toFixed(2)} km<br>
      Parking space: ${lot.available_spaces} / ${lot.total_spaces}<br>
      Score: ${lot.score}<br>
      <button onclick="viewParking(${index})">View</button>
      <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>
      <button onclick="togglePredictionSection(${index})">Prediction</button>

      <!-- Forecast area (hidden by default), which will be displayed after clicking the button -->
      <div id="prediction-section-${index}" class="prediction-section" style="display: none; margin-top: 10px;">
        
        <!-- Drop-down box (from 15 minutes to 24 hours, every 15 minutes) -->
        <label>Select Interval:</label>
        <select id="prediction-select-${index}">
          ${generateTimeOptions()} 
        </select>
        <button onclick="submitPrediction(${lot.lot_id}, ${index})">OK</button>

        <div id="prediction-result-${index}" class="prediction-result" style="margin-top: 10px; color: #333;"></div>
      </div>
    `;
        parkingList.appendChild(item);
    });
}

/**
 * Opens Google Maps directions in a new tab.
 * Attempts to get the user's current location via HTML5 geolocation
 * for a more accurate origin.
 * @param {number} lat - Destination latitude
 * @param {number} lng - Destination longitude
 */
function navigateToParking(lat, lng) {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                const userLat = currentLocation.lat;
                const userLng = currentLocation.lng;
                console.log("User's current location:", userLat, userLng);
                const url = `https://www.google.com/maps/dir/?api=1&origin=${userLat},${userLng}&destination=${lat},${lng}&travelmode=driving`;
                window.open(url, "_blank");
            },
            error => {
                console.error("Unable to obtain the user's current location:", error);

                // Fall back to just the destination if we cannot get origin
                const url = `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}&travelmode=driving`;
                window.open(url, "_blank");
            }
        );
    } else {
        console.error("Browser does not support geolocation.");
        const url = `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}&travelmode=driving`;
        window.open(url, "_blank");
    }
}

function predictedParkingSpaces(lat, lng) {
    console.log("Predicting parking spaces for:", lat, lng);
    // Implement your prediction logic here
    // For example, you can call a prediction API or use a machine learning model
    // to estimate the number of available parking spaces at a given location.
}


/**
 * Focus the map on a specific parking lot, open its InfoWindow,
 * and close any previously opened InfoWindow.
 * @param {number} index - The index of the parking lot in parkingData
 */
function viewParking(index) {
    const lot = parkingData[index];
    console.log(`Focusing on parking lot: ${lot.name}`);

    console.log(`infoWindow: ${lot.infoWindow}`);

    // Close the previously opened info window if any
    if (currentInfoWindow) {
        currentInfoWindow.close();
    }

    // Center and zoom in on the selected parking lot
    map.setCenter({ lat: lot.latitude, lng: lot.longitude });
    map.setZoom(16);

    // Open its InfoWindow
    setInfoWindow(lot);
    currentInfoWindow.open(map, lot.marker);
}

function generateTimeOptions() {
    let html = '';
    for (let i = 15; i <= 1440; i += 15) {
        html += `<option value="${i}" ${i === 60 ? 'selected' : ''}>${i} minutes</option>`;
    }
    return html;
}

function togglePredictionSection(index) {
    const section = document.getElementById(`prediction-section-${index}`);
    if (!section) return;
    if (section.style.display === 'none') {
        section.style.display = 'block';
    } else {
        section.style.display = 'none';
    }
}

function submitPrediction(lot_id, index) {
    // 1) Get the number of minutes selected by the user
    const selectEl = document.getElementById(`prediction-select-${index}`);
    const chosenMinutes = parseInt(selectEl.value, 10);

    // 2) Calculate "Predicted Execution Time" and "Predicted Downtime"
    const now = new Date();
    const nowStr = formatDateTime(now);
    const future = new Date(now.getTime() + chosenMinutes * 60000);
    const futureStr = formatDateTime(future);

    // 3) Demo request to the prediction API
    // { predicted_spaces: 10, predicted_score: 0.85, message: "ok" }
    fetch(`${window.API_CONFIG.API_BASE_URL}/api/predict?lot_id=${lot_id}&predition_time=${nowStr}&lopp_time=${chosenMinutes / 15}`)
        .then(response => response.json())
        .then(data => {
            // 4) Display the data returned by the backend + the time calculated above to the page
            const resultEl = document.getElementById(`prediction-result-${index}`);
            resultEl.innerHTML = `
              <p><strong>Prediction execution time:</strong> ${nowStr}</p>
              <p><strong>Prediction parking time:</strong> ${futureStr}</p>
              <p><strong>Vehicles increased:</strong> ${data}</p>
            `;
        })
        .catch(err => {
            console.error('Prediction request error:', err);
            alert('Prediction failed: ' + err);
        });

}

function formatDateTime(dateObj) {
    const year = dateObj.getFullYear();
    const month = String(dateObj.getMonth() + 1).padStart(2, '0');
    const day = String(dateObj.getDate()).padStart(2, '0');
    const hour = String(dateObj.getHours()).padStart(2, '0');
    const minute = String(dateObj.getMinutes()).padStart(2, '0');
    const second = String(dateObj.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}




// Expose functions to the global scope if needed
window.loadParkingData = loadParkingData;
window.navigateToParking = navigateToParking;
window.predictedParkingSpaces = predictedParkingSpaces;
window.viewParking = viewParking;
