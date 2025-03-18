const DEFAULT_LAT = -36.7261;
const DEFAULT_LNG = 174.7094;

let map;
let userMarker;
let currentInfoWindow;

// Store the user's current location (lat, lng, address)
let currentLocation = {
    lat: null,
    lng: null,
    address: ""
};

window.currentLocation = {
    lat: DEFAULT_LAT,
    lng: DEFAULT_LNG,
    address: ""
};



// ========== Google Map Initialization ==========
window.initMap = function () {
    // If you need to restore a previously saved currentLocation from localStorage
    loadCurrentLocationFromStorage();

    if (DEBUG_MODE) {
        // DEBUG MODE: get a test location from a local JSON file
        fetch("json/location.json")
            .then(response => response.json())
            .then(({ latitude, longitude }) => initializeMap(latitude, longitude))
            .catch(() => initializeMap(DEFAULT_LAT, DEFAULT_LNG));
    } else {
        // Normal mode: use the browser's geolocation
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                ({ coords }) => initializeMap(coords.latitude, coords.longitude),
                () => initializeMap(DEFAULT_LAT, DEFAULT_LNG)
            );
        } else {
            initializeMap(DEFAULT_LAT, DEFAULT_LNG);
        }
    }
};

async function initializeMap(lat, lng) {
    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat, lng },
        zoom: 14
    });

    userMarker = new google.maps.Marker({
        position: { lat, lng },
        map: map,
        icon: "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
        title: "Your Location"
    });
    // init compoments 
    currentInfoWindow = new google.maps.InfoWindow();

    // Initialize address autocomplete
    initAutocomplete();

    // Set and fetch the address for the current location
    await setCurrentLocation(lat, lng);

    // Update the UI text with the address
    updateCurrentLocationText();

    // Save the location (optional)
    saveUserLocation(lat, lng);

    // Load parking data (from parking.js or elsewhere)
    loadParkingData();
}

// ========== Address Autocomplete Setup ==========
function initAutocomplete() {
    const input = document.getElementById("address-input");
    const autocomplete = new google.maps.places.Autocomplete(input, {
        componentRestrictions: { country: "nz" },
    });

    autocomplete.addListener("place_changed", () => {
        const place = autocomplete.getPlace();
        if (place.geometry) {
            const { lat, lng } = place.geometry.location;
            setUserLocation(lat(), lng());
        }
    });
}

// ========== Set User Location and Save ==========

/**
 * Updates map center/marker and fetches address information.
 */
async function setUserLocation(lat, lng) {
    map.setCenter({ lat, lng });
    map.setZoom(14);

    if (userMarker) {
        userMarker.setPosition({ lat, lng });
    } else {
        userMarker = new google.maps.Marker({
            position: { lat, lng },
            map: map,
            icon: "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
            title: "Your Location"
        });
    }

    // Fetch address and update currentLocation
    await setCurrentLocation(lat, lng);

    // Update the UI text with the address
    updateCurrentLocationText();

    // Save location details
    saveUserLocation(lat, lng);

    // Reload parking data, if needed
    loadParkingData();
}



/**
 * Fetches the address from Google Geocoding and updates currentLocation.
 */
async function setCurrentLocation(lat, lng) {
    try {
        const address = await getAddressFromLatLng(lat, lng);
        currentLocation.lat = lat;
        currentLocation.lng = lng;
        currentLocation.address = address;
    } catch (error) {
        console.error("Failed to get address:", error);
        currentLocation.lat = lat;
        currentLocation.lng = lng;
        currentLocation.address = "";
    }
}

// ========== Update the UI with the current address ==========
function updateCurrentLocationText() {
    const locationTextElem = document.getElementById("current-location-text");
    locationTextElem.innerHTML = `<strong>Current Location:</strong> 📍 ${currentLocation.address}`;
}

async function getGoogleMapsApiKey() {
    const response = await fetch("/config", {
        method: "GET",
        headers: {
            "Authorization": "your-secret-token"
        }
    });
    const data = await response.json();
    return data.google_maps_api_key;
}

// ========== Get address by lat/lng (reverse geocoding) ==========
async function getAddressFromLatLng(lat, lng) {
    try {
        const apiKey = await getGoogleMapsApiKey();
        const response = await fetch(`https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lng}&language=en&key=${apiKey}`);
        const data = await response.json();
        if (data.results && data.results[0]) {
            return data.results[0].formatted_address;
        }
        return `Lat: ${lat}, Lng: ${lng}`;
    } catch (error) {
        console.error("Failed to get address:", error);
        return `Lat: ${lat}, Lng: ${lng}`;
    }
}

// ========== Save the current location to localStorage ==========
async function saveUserLocation(lat, lng) {
    try {
        const address = await getAddressFromLatLng(lat, lng);

        currentLocation.lat = lat;
        currentLocation.lng = lng;
        currentLocation.address = address;

        localStorage.setItem("currentLocation", JSON.stringify(currentLocation));
    } catch (error) {
        console.error("There was an error saving the address:", error);
    }
}

// ========== Load current location from localStorage ==========
function loadCurrentLocationFromStorage() {
    const data = localStorage.getItem("currentLocation");
    if (data) {
        currentLocation = JSON.parse(data);
    }
}

// ========== Event Listeners ==========

document.getElementById("search-btn").addEventListener("click", () => {
    loadParkingData();
});

document.getElementById("current-location-btn").addEventListener("click", () => {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            ({ coords }) => setUserLocation(coords.latitude, coords.longitude),
            () => alert("Failed to retrieve location. Please check your browser settings.")
        );
    } else {
        alert("Geolocation is not supported by your browser.");
    }
});
