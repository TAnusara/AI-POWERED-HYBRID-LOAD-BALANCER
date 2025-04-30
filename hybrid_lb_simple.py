import numpy as np
from flask import Flask, jsonify, render_template
import requests
import psutil
import tensorflow as tf
import subprocess
from threading import Thread
import time
from werkzeug.serving import make_server
import math

app = Flask(__name__)

# ======== Configuration ========
SERVERS = ["http://localhost:8081", "http://localhost:8082"]
MODEL_PATH = "lb_model.h5"
NGINX_CONFIG = "/etc/nginx/sites-available/loadbalancer"
current_server = 0

# ======== Mathematical Load Metrics ========
def calculate_integral_load(history_window=5):
    """Calculate integral of system metrics over time"""
    metrics = []
    for _ in range(history_window):
        metrics.append([
            psutil.cpu_percent(),
            psutil.virtual_memory().percent,
            psutil.net_io_counters().bytes_sent,
            psutil.net_io_counters().bytes_recv
        ])
        time.sleep(0.5)
    
    # Trapezoidal integration
    integral = np.trapz(metrics, axis=0)
    return integral / history_window  # Normalized

# ======== AI Model ========
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    print("Training initial model...")
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=10, verbose=0)
    model.save(MODEL_PATH)

# ======== NGINX Management ========
def configure_nginx_fallback():
    config = f"""
    server {{
        listen 80;
        server_name localhost;
        
        location / {{
            proxy_pass http://localhost:5000;
            proxy_intercept_errors on;
            error_page 502 503 504 = @fallback;
        }}
        
        location @fallback {{
            proxy_pass http://localhost:8080;
        }}
    }}
    """
    with open(NGINX_CONFIG, "w") as f:
        f.write(config)
    subprocess.run(["sudo", "nginx", "-s", "reload"])

# ======== Routes ========
@app.route('/')
def dashboard():
    return render_template('dashboard.html', servers=SERVERS)

@app.route('/api/route')
def route_request():
    # Mathematical decision component
    integral_load = calculate_integral_load()
    math_score = 0.5 + (0.1 * math.log(1 + integral_load[0]))  # CPU-weighted
    
    # AI decision component
    ai_input = np.array([[
        psutil.cpu_percent(),
        psutil.virtual_memory().percent,
        psutil.net_io_counters().bytes_sent,
        psutil.net_io_counters().bytes_recv
    ]], dtype=np.float32)
    ai_score = model.predict(ai_input)[0][0]
    
    # Hybrid decision (60% AI, 40% Math)
    combined_score = 0.6 * ai_score + 0.4 * math_score
    selected_server = SERVERS[int(combined_score > 0.5)]
    
    try:
        response = requests.get(selected_server, timeout=2)
        return jsonify({
            "server": selected_server,
            "ai_score": float(ai_score),
            "math_score": float(math_score),
            "combined_score": float(combined_score),
            "system_metrics": dict(zip(
                ["cpu", "memory", "network_sent", "network_recv"],
                ai_input[0].tolist()
            ))
        })
    except:
        return jsonify({"error": "All servers down"}), 502

# ======== Monitoring Endpoints ========
@app.route('/api/metrics')
def get_metrics():
    return jsonify({
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "network": psutil.net_io_counters()._asdict()
    })

# ======== Main Execution ========
if __name__ == '__main__':
    # Setup NGINX fallback
    configure_nginx_fallback()
    
    # Start background monitor
    Thread(target=monitor_system, daemon=True).start()
    
    # Start Flask server
    server = make_server('0.0.0.0', 5000, app)
    print("""
    Hybrid Load Balancer Running!
    ----------------------------
    Dashboard: http://localhost:5000
    API Routes:
      - /api/route
      - /api/metrics
    NGINX Fallback: http://localhost:80
    """)
    server.serve_forever()
