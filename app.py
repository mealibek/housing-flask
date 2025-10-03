
from flask import Flask, request, render_template, jsonify
from ml.model_loader import FEATURE_ORDER
from ml.preprocess import preprocess_input
from ml.model_loader import model

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route("/", methods=["GET"])
def index():
    # render a simple HTML form where users can input features
    return render_template("index.html", features=FEATURE_ORDER)

@app.route("/predict", methods=["POST"])
def predict_form():
    # handle form submission: form values are strings
    try:
        payload = { feat: request.form.get(feat) for feat in FEATURE_ORDER }
        X = preprocess_input(payload)
        pred = model.predict(X)[0]
        return render_template("result.html", prediction=pred, inputs=payload)
    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route("/api/predict", methods=["POST"])
def predict_api():
    # Accept JSON body with { feature_name: value, ... }
    if not request.is_json:
        return jsonify({"error":"Request must be JSON"}), 400
    payload = request.get_json()
    try:
        X = preprocess_input(payload)
        pred = model.predict(X)[0]
        return jsonify({
            "prediction": float(pred),
            "inputs": payload
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
