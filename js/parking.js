// parking.js

// Global array to store all parking lot data loaded from JSON or an API
let parkingData = [];

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
    console.log("DEBUG_MODE:", DEBUG_MODE);

    const dataSource = DEBUG_MODE ? "json/parking_data.json" : "/api/get-parking-data";
    console.log("Current location in parking.js:", currentLocation);

    fetch(dataSource)
        .then(response => response.json())
        .then(data => {
            // Compute the distance from the user's currentLocation to each parking lot
            parkingData = data.map(lot => ({
                ...lot,
                distance: getDistance(
                    currentLocation.lat,
                    currentLocation.lng,
                    lot.latitude,
                    lot.longitude
                )
            }));

            // Sort parking lots from closest to farthest
            parkingData.sort((a, b) => a.distance - b.distance);

            // Create a Marker and InfoWindow for each parking lot
            parkingData.forEach(lot => {
                const marker = new google.maps.Marker({
                    position: { lat: lot.latitude, lng: lot.longitude },
                    map: map, // 'map' should be global from map.js
                    title: lot.name
                });

                const infoWindow = new google.maps.InfoWindow({
                    content: `
            <h3>${lot.name}</h3>
            <p>Available: ${lot.available_spaces} / ${lot.total_spaces}</p>
            <p>Price: ${lot.price_per_hour}</p>
            <p>Distance: ${lot.distance.toFixed(2)} km</p>
            <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>
          `
                });

                marker.addListener("click", () => {
                    infoWindow.open(map, marker);
                });

                lot.marker = marker;
                lot.infoWindow = infoWindow;
            });

            // Update the list in the side panel
            updateParkingList();
        })
        .catch(error => {
            console.error("Failed to load parking data:", error);
        });
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

    parkingData.forEach((lot, index) => {
        const item = document.createElement('li');
        item.classList.add('parking-item');
        item.innerHTML = `
      <strong>${lot.name}</strong><br>
      Distance: ${lot.distance.toFixed(2)} km<br>
      Parking space: ${lot.available_spaces} / ${lot.total_spaces}<br>
      Price: ${lot.price_per_hour}<br>
      <button onclick="viewParking(${index})">View</button>
      <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>
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
                const userLat = position.coords.latitude;
                const userLng = position.coords.longitude;

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

// Keep track of the currently opened InfoWindow so we can close it when viewing another one
let currentInfoWindow = null;

/**
 * Focus the map on a specific parking lot, open its InfoWindow,
 * and close any previously opened InfoWindow.
 * @param {number} index - The index of the parking lot in parkingData
 */
function viewParking(index) {
    const lot = parkingData[index];
    console.log(`Focusing on parking lot: ${lot.name}`);

    // Close the previously opened info window if any
    if (currentInfoWindow) {
        currentInfoWindow.close();
    }

    // Center and zoom in on the selected parking lot
    map.setCenter({ lat: lot.latitude, lng: lot.longitude });
    map.setZoom(16);

    // Open its InfoWindow
    lot.infoWindow.open(map, lot.marker);
    currentInfoWindow = lot.infoWindow;
}

// Expose functions to the global scope if needed
window.loadParkingData = loadParkingData;
window.navigateToParking = navigateToParking;
window.viewParking = viewParking;
