const API_KEY = "AIzaSyD2r8GYegCiPsXwkk-fxELUhpr4gUgb1Dk";  // Here we will use .env method to save it later

const DEFAULT_LAT = -36.7261;
const DEFAULT_LNG = 174.7094;

// Initialize Google Maps
let map, userLat, userLng, userMarker;


window.initMap = function () {
    if (DEBUG_MODE) {
        console.log("DEBUG_MODE: Using location.json for user location");

        fetch("json/location.json")
            .then(response => response.json())
            .then(locationData => {
                userLat = locationData.latitude;
                userLng = locationData.longitude;
                initializeMap(userLat, userLng);
            })
            .catch(error => {
                console.error("Failed to load location.json, using default coordinates:", error);
                initializeMap(DEFAULT_LAT, DEFAULT_LNG);
            });

    } else {
        console.log("DEBUG_MODE: Using real GPS location");

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(position => {
                userLat = position.coords.latitude;
                userLng = position.coords.longitude;
                initializeMap(userLat, userLng);
            }, error => {
                console.error("Failed to get GPS location, using default coordinates:", error);
                initializeMap(DEFAULT_LAT, DEFAULT_LNG);
            });
        } else {
            console.error("Geolocation not supported, using default coordinates.");
            initializeMap(DEFAULT_LAT, DEFAULT_LNG);
        }
    }
};

// Unified methods of initializing maps to reduce duplicate code
function initializeMap(lat, lng) {
    userLat = lat;
    userLng = lng;

    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: userLat, lng: userLng },
        zoom: 14
    });

    userMarker = new google.maps.Marker({
        position: { lat: userLat, lng: userLng },
        map: map,
        icon: "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        title: "Your Location"
    });

    loadParkingData();
}


// Dynamic loading of Google Maps API
const script = document.createElement("script");
script.src = "https://maps.googleapis.com/maps/api/js?key=" + API_KEY + "&libraries=places&callback=initMap";
script.async = true;
script.defer = true;
document.head.appendChild(script);

