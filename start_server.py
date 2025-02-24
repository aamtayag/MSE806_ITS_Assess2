import http.server
import socket
import webbrowser
import threading

PORT = 8000

# Get the IP address of the machine (for multiple platforms)
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to external network to obtain native IP
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return ip


def start_server():
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Serving at http://{get_local_ip()}:{PORT}")
        print("Press Ctrl+C to stop the server.")
        httpd.serve_forever()


def open_browser():
    url = f"http://{get_local_ip()}:{PORT}"
    print(f"Opening {url} in default browser...")
    webbrowser.open(url)

# Start the server and browser (use multithreading to ensure that it will not block)
if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    open_browser()
