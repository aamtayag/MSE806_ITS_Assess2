let parkingData = [];


// Calculate the distance between two points
function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// Load parking data from JSON file or API
function loadParkingData() {
    console.log("DEBUG_MODE:", DEBUG_MODE);
    const dataSource = true ? "parking_data.json" : "/api/get-parking-data";

    fetch(dataSource)
        .then(response => response.json())
        .then(data => {
            parkingData = data.map(lot => ({
                ...lot,
                distance: getDistance(userLat, userLng, lot.latitude, lot.longitude)
            }));

            parkingData.sort((a, b) => a.distance - b.distance);

            parkingData.forEach(lot => {
                const marker = new google.maps.Marker({
                    position: { lat: lot.latitude, lng: lot.longitude },
                    map: map,
                    title: lot.name
                });

                const infoWindow = new google.maps.InfoWindow({
                    content: `<h3>${lot.name}</h3>
                              <p>Available: ${lot.available_spaces} / ${lot.total_spaces}</p>
                              <p>Price: ${lot.price_per_hour}</p>
                              <p>Distance: ${lot.distance.toFixed(2)} km</p>
                              <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>`
                });

                marker.addListener("click", () => {
                    infoWindow.open(map, marker);
                });

                lot.marker = marker;
                lot.infoWindow = infoWindow;
            });

            updateParkingList();
        })
        .catch(error => {
            console.error("Failed to load JSON data:", error);
        });
}

// Update the parking lot list
function updateParkingList() {
    const parkingList = document.getElementById('parking-items');
    parkingList.innerHTML = "";

    parkingData.forEach((lot, index) => {
        const item = document.createElement('li');
        item.classList.add('parking-item');
        item.innerHTML = `<strong>${lot.name}</strong><br>
                          Distance: ${lot.distance.toFixed(2)} km<br>
                          Parking space: ${lot.available_spaces} / ${lot.total_spaces}<br>
                          Price: ${lot.price_per_hour}<br>
                          <button onclick="viewParking(${index})">View</button>
                          <button onclick="navigateToParking(${lot.latitude}, ${lot.longitude})">Navigate</button>`;
        parkingList.appendChild(item);
    });
}

// Let the Google Maps API recognize initMap
window.loadParkingData = loadParkingData;
