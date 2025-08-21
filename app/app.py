from flask import Flask, request, jsonify
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from Sharp_demo import run_demo
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

api_call_counter = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method']
)

response_time_histogram = Histogram(
    'api_response_time_seconds',
    'Histogram of API response times',
    ['endpoint', 'method']
)

# Simulated dependencies
database_connected = True
model_loaded = True

@app.route("/health/live", methods=["GET"])
def health_live():
    """
    Liveness check: Is the app running?
    Return 200 if yes, 503 if not.
    """
    return jsonify({"status": "alive"}), 200

@app.route("/health/ready", methods=["GET"])
def health_ready():
    """
    Readiness check: Are dependencies working?
    Example: DB connection, model loaded.
    """
    if database_connected and model_loaded:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "not ready"}), 503

@app.route("/health/detailed", methods=["GET"])
def health_detailed():
    """
    Detailed health: Shows status of every dependency.
    """
    details = {
        "app_status": "running",
        "timestamp": time.time(),
        "database_connected": database_connected,
        "model_loaded": model_loaded,
        # "random_latency_ms": random.randint(5, 50)  # Example metric
    }

    overall_status = 200 if all([database_connected, model_loaded]) else 503
    return jsonify(details), overall_status

@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def track_metrics(response):
    if request.endpoint != 'metrics':  # avoid counting metrics calls
        resp_time = time.time() - request.start_time
        api_call_counter.labels(request.path, request.method).inc()
        response_time_histogram.labels(request.path, request.method).observe(resp_time)
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        # "task_id": sharp_mgr.task_id,
        # "model_trained": sharp_mgr.model is not None,
    })

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route("/predict", methods=["POST"])
def predict():

    results = run_demo()
    return results

# Optional endpoint to run the original demo script if available
@app.route("/run-demo", methods=["GET"])
def run_demo_endpoint():
    if not HAS_RUN_DEMO:
        return jsonify({"error": "run_demo not available; Sharp_demo.py not importable"}), 500
    try:
        results = run_demo()
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": f"Demo failed: {e}"}), 500

if __name__ == "__main__":
    # Run on localhost:8001
    app.run(host="127.0.0.1", port=8001, debug=True)

