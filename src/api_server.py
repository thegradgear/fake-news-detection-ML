from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from src.model_evaluation import ModelEvaluator

app = Flask(__name__)
CORS(app)

# Load model and evaluator
evaluator = ModelEvaluator()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Fake News Detection API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict if news is fake or real",
            "/health": "GET - Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        if len(text.strip()) < 10:
            return jsonify({"error": "Text too short for reliable prediction"}), 400
        
        # Make prediction
        result = evaluator.predict_single(text)
        
        return jsonify({
            "prediction": result['prediction'],
            "confidence": float(result['confidence']),
            "probabilities": {
                "fake": float(result['probabilities']['fake']),
                "real": float(result['probabilities']['real'])
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        results = []
        for text in texts:
            if len(text.strip()) >= 10:
                result = evaluator.predict_single(text)
                results.append(result)
            else:
                results.append({"error": "Text too short"})
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)