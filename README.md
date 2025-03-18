# Parking Lot Management System

## Overview
This is a web-based Parking Lot Management System that provides functionalities for managing parking lots, vehicles, and AI-based predictions for available spaces.

## Features
- Manage parking lots (Create, Read, Update, Delete)
- AI-based prediction of available spaces
- Google Maps integration for geolocation services
- User-friendly UI with Bootstrap
- Server running based on HTTPS

## Git Clone
```sh
git clone https://github.com/aamtayag/MSE806_ITS_Assess2
```


## Installation
### Prerequisites
- Python 3.9
- SQLite

### Setup Instructions
### 1. Create a Python Virtual Environment
```sh
python -m venv itsenv
source venv/bin/activate  

# On Windows use: 
itsenv\Scripts\activate
```

### 2. Install Backend Dependencies
```sh
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add the following variables:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```
### 3.1(Optional). Update Authorization Token
```
In map.js and index.html, Simple token verification is used, and you can adjust it according to your needs

"Authorization": "your-secret-token"
```

## Google Maps Integration
Ensure you have a valid **Google Maps API Key** and update the `.env` file accordingly. The system uses Google Maps for:
- Displaying parking locations
- Reverse geocoding to fetch addresses

## HTTPS Configuration
To enable **HTTPS** for your local development server, you need to generate an **SSL certificate**. We will use `mkcert`, a simple tool for creating locally trusted certificates.

### **4.1 Install `mkcert`**
#### **For macOS (Homebrew)**
```sh
brew install mkcert
mkcert -install
```

#### **For Windows (Chocolatey)**
```sh
choco install mkcert
mkcert -install
```

#### **For Linux (Debian/Ubuntu)**
```sh
sudo apt install libnss3-tools
wget -qO - https://github.com/FiloSottile/mkcert/releases/latest/download/mkcert-v1.4.4-linux-amd64 | sudo tee /usr/local/bin/mkcert > /dev/null
sudo chmod +x /usr/local/bin/mkcert
mkcert -install
```

### **4.2 Verify Installation**
Run the following command to check if `mkcert` is installed correctly:
```sh
mkcert -help
```
You should see a list of available commands.

### **4.3 Install Root CA Certificate**
`mkcert` installs a local **Certificate Authority (CA)** that allows your browser to trust self-signed certificates:
```sh
mkcert -install
```
If successful, you should see:
```
Created a new local CA
```

### **4.4 Generate SSL Certificates**
Now, generate your **SSL certificate and key**:
```sh
mkcert -key-file key.pem -cert-file cert.pem 127.0.0.1 localhost $(hostname)
```
This will create two files:
- **`cert.pem`** → The SSL certificate
- **`key.pem`** → The private key

### **4.5 Confirm Certificate Files**
Run:
```sh
ls -l cert.pem key.pem
```
If you see two valid files, the setup is complete!

### 5. Start Server
```sh
python api_server.py
```

## Manager Page
```sh
IpAddress:{port}/manager.html

for example:
index.html =====>   https://192.168.1.100:8000
manager.html =====>   https://192.168.1.100:8000/manager.html
```


## API Endpoints
### Parking Lot Endpoints
- `GET /parking_lots` - Get all parking lots
- `POST /parking_lot` - Create a new parking lot
- `PUT /parking_lot/<id>` - Update a parking lot
- `DELETE /parking_lot/<id>` - Delete a parking lot

### Prediction Endpoints
- `GET /predictions` - Get all predictions
- `POST /prediction` - Create a new prediction
- `PUT /prediction/<id>` - Update a prediction
- `DELETE /prediction/<id>` - Delete a prediction

## Google Maps Integration
Ensure you have a valid **Google Maps API Key** and update the `.env` file accordingly. The system uses Google Maps for:
- Displaying parking locations
- Reverse geocoding to fetch addresses

## Troubleshooting
### Common Issues
1. **Database Connection Error**
   - Ensure the database service is running
   - Verify `DATABASE_URL` in `.env` is correctly set

2. **Google Maps Not Loading**
   - Check if `GOOGLE_MAPS_API_KEY` is valid
   - Verify that Maps JavaScript API is enabled in Google Cloud Console

3. **Frontend Not Loading**
   - Ensure `python all package` has completed successfully
   - Check if the backend server is running

## Contributors
- Wen Liang

## License
This project is licensed under the MIT License.
