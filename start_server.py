import http.server
import ssl
import os
import socket
import webbrowser
import threading

PORT = 8000

# Get absolute path to the 'cert' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(BASE_DIR, "cert")

# SSL Certificate and Key Paths
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "key.pem")


def get_local_ip():
    """Retrieve the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return ip


def start_server():
    """Start HTTPS server"""
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler)

    # Use modern SSL/TLS configuration
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)

    # Wrap the server socket with SSL
    server.socket = context.wrap_socket(server.socket, server_side=True)

    print(f"Serving at https://{get_local_ip()}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


def open_browser():
    """Automatically open the default browser"""
    url = f"https://{get_local_ip()}:{PORT}"
    print(f"Opening {url} in default browser...")
    webbrowser.open(url)


if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    open_browser()
    input("Press Enter to exit...\n")  # Prevent script from exiting immediately
