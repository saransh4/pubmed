from flask import Flask, request, jsonify
import time
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BioGptForCausalLM
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ------------------ Prometheus Metrics ------------------
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

# ------------------ Dependencies ------------------
database_connected = True
model_cache = {}   # cache to avoid reloading models repeatedly

PROMPT_TMPL = (
    "You are a biomedical expert.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

# ------------------ Utility Functions ------------------
def load_model(model_name: str):
    """Load base BioGPT or LoRA model depending on model_name."""
    if model_name in model_cache:
        return model_cache[model_name]

    if "biogpt" in model_name.lower() and "lora" not in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BioGptForCausalLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_cache[model_name] = (model, tokenizer)
    return model, tokenizer

def generate_answer(model, tokenizer, question: str, context: str, max_new_tokens: int = 100):
    """Generate an answer from a given model."""
    prompt = PROMPT_TMPL.format(question=question, context=context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

# ------------------ Health Endpoints ------------------
@app.route("/health/live", methods=["GET"])
def health_live():
    return jsonify({"status": "alive"}), 200

@app.route("/health/ready", methods=["GET"])
def health_ready():
    if database_connected:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "not ready"}), 503

@app.route("/health/detailed", methods=["GET"])
def health_detailed():
    details = {
        "app_status": "running",
        "timestamp": time.time(),
        "database_connected": database_connected,
        "cached_models": list(model_cache.keys())
    }
    return jsonify(details), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# ------------------ Prediction Endpoint ------------------
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def track_metrics(response):
    if request.endpoint != 'metrics':
        resp_time = time.time() - request.start_time
        api_call_counter.labels(request.path, request.method).inc()
        response_time_histogram.labels(request.path, request.method).observe(resp_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        question = data.get("question")
        context = data.get("context", "")

        if not model_name or not question:
            return jsonify({"error": "model_name and question are required"}), 400

        model, tokenizer = load_model(model_name)
        answer = generate_answer(model, tokenizer, question, context)

        return jsonify({
            "model_name": model_name,
            "question": question,
            "context": context,
            "answer": answer
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
