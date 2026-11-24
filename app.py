"""
app.py
Flask Web Application for Fine-Tuned DistilBERT Sentiment Analysis & Review Intelligence
Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)
MIT Academy of Engineering (MITAOE) • Department of Data Science

Provides:
- Interactive dashboard UI for single review analysis and batch CSV inference.
- Real-time confidence scores, polarity gauges, token counts, and class probabilities.
- Model evaluation metrics comparison (Baseline vs. Fine-Tuned).
- Interactive Chart.js visualizations (Sentiment distribution, Metrics comparison, Training dynamics).
- Input validation, loading states, and error handling.
"""

import os
import json
import io
import csv

try:
    import pandas as pd
except ImportError:
    pd = None

from flask import Flask, render_template, request, jsonify, send_file
from inference import SentimentClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')

# Initialize inference engine lazily
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        try:
            classifier = SentimentClassifier()
        except Exception as e:
            print(f"[Warning] Could not load fine-tuned model immediately: {e}")
    return classifier


@app.route('/')
def index():
    """Renders the main Review Intelligence Dashboard."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for single review inference."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'status': 'error', 'message': 'Please provide non-empty review text.'}), 400

        engine = get_classifier()
        if engine is None:
            return jsonify({'status': 'error', 'message': 'Model is not loaded yet. Ensure train.py has completed.'}), 503

        result = engine.predict(text)
        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Inference error: {str(e)}'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch review inference via CSV file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded. Please select a CSV file.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'Invalid file format. Only CSV files are supported.'}), 400

        # Read CSV with pandas or fallback csv reader
        try:
            if pd is not None:
                df = pd.read_csv(file)
                if df.empty:
                    return jsonify({'status': 'error', 'message': 'The uploaded CSV file is empty.'}), 400
            else:
                stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
                reader = csv.DictReader(stream)
                df = list(reader)
                if not df:
                    return jsonify({'status': 'error', 'message': 'The uploaded CSV file is empty.'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to parse CSV file: {str(e)}'}), 400

        engine = get_classifier()
        if engine is None:
            return jsonify({'status': 'error', 'message': 'Model is not loaded yet.'}), 503

        # Run analysis (handles candidate review columns)
        try:
            analysis = engine.analyze_dataframe(df)
        except ValueError as ve:
            return jsonify({'status': 'error', 'message': str(ve)}), 400

        return jsonify({'status': 'success', 'data': analysis})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Batch processing error: {str(e)}'}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns actual evaluation metrics and comparison data from results/metrics.json."""
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    history_path = os.path.join(RESULTS_DIR, 'training_history.json')

    metrics_data = {}
    history_data = {}

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)

    return jsonify({
        'status': 'success',
        'metrics': metrics_data,
        'training_history': history_data
    })


@app.route('/api/download-sample-csv', methods=['GET'])
def download_sample_csv():
    """Provides sample CSV for viva demonstration."""
    sample_path = os.path.join(DATASET_DIR, 'sample_test_reviews.csv')
    if os.path.exists(sample_path):
        return send_file(sample_path, as_attachment=True, download_name='sample_test_reviews.csv', mimetype='text/csv')
    else:
        # Generate inline fallback if file not yet created
        csv_data = (
            "review_id,text\n"
            "1,\"An incredible cinematic achievement with superb acting and direction.\"\n"
            "2,\"Terrible script, dull pacing, and completely unconvincing characters.\"\n"
            "3,\"Visually stunning cinematography, though the narrative drags in the middle.\"\n"
            "4,\"A delightfully hilarious comedy that had the entire theatre in stitches.\"\n"
            "5,\"A gripping psychological noir with razor-sharp dialogue and a haunting climax.\"\n"
        )
        buffer = io.BytesIO(csv_data.encode("utf-8"))
        return send_file(buffer, as_attachment=True, download_name='sample_reviews.csv', mimetype='text/csv')


@app.route('/api/health', methods=['GET'])
def health():
    """System health check and hardware diagnostics."""
    engine = get_classifier()
    model_loaded = engine is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'candidate': 'Mahesh Kadam (Roll 51, PRN: 202401120074)',
        'model_path': engine.model_dir if model_loaded else None,
        'compute_device': str(engine.device) if model_loaded else 'unknown'
    })


if __name__ == '__main__':
    print("[Server] Starting Review Intelligence Flask Service...")
    print("[Server] Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)")
    get_classifier()
    app.run(host='0.0.0.0', port=5000, debug=False)
