#!/usr/bin/env python3
'''
Local F1 Web App Demo Server
'''

import http.server
import socketserver
import webbrowser
import os
import threading
import time

def start_demo_server():
    PORT = 8000
    
    # Change to docs directory
    os.chdir('docs')
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers for local development
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 F1 Web App Demo Server running at: http://localhost:{PORT}")
        print("🏎️ Opening web browser...")
        
        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{PORT}')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        print("\n🔧 Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    start_demo_server()
