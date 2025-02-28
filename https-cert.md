# **Step 1: Install `mkcert` (Generate HTTPS Certificates)**

To enable **HTTPS** for your local development server, you need to generate an **SSL certificate**. We will use `mkcert`, a simple tool for creating locally trusted certificates.

---

## **1.1 Install `mkcert`**
### **For macOS (Homebrew)**
```sh
brew install mkcert
mkcert -install
```

### **For Windows (Chocolatey)**
```sh
choco install mkcert
mkcert -install
```

### **For Linux (Debian/Ubuntu)**
```sh
sudo apt install libnss3-tools
wget -qO - https://github.com/FiloSottile/mkcert/releases/latest/download/mkcert-v1.4.4-linux-amd64 | sudo tee /usr/local/bin/mkcert > /dev/null
sudo chmod +x /usr/local/bin/mkcert
mkcert -install
```

---

## **1.2 Verify Installation**
Run the following command to check if `mkcert` is installed correctly:
```sh
mkcert -help
```
You should see a list of available commands.

---

## **1.3 Install Root CA Certificate**
`mkcert` installs a local **Certificate Authority (CA)** that allows your browser to trust self-signed certificates:
```sh
mkcert -install
```
If successful, you should see:
```
Created a new local CA
```

---

## **1.4 Generate SSL Certificates**
Now, generate your **SSL certificate and key**:
```sh
mkcert -key-file key.pem -cert-file cert.pem 127.0.0.1 localhost $(hostname)
```
This will create two files:
- **`cert.pem`** → The SSL certificate
- **`key.pem`** → The private key

---

## **1.5 Confirm Certificate Files**
Run:
```sh
ls -l cert.pem key.pem
```
If you see two valid files, the setup is complete!
