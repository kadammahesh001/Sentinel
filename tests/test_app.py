"""
test_app.py
Comprehensive Automated Unit and Integration Tests for DistilBERT Review Intelligence
Candidate: Mahesh Kadam (Roll 51, PRN: 202401120074)

Tests:
1. Model loading and inference initialization.
2. Positive review prediction and confidence score.
3. Negative review prediction and confidence score.
4. Empty and whitespace input validation.
5. Long text sequence handling and truncation.
6. Batch prediction on list of strings.
7. CSV file upload endpoint (valid CSV).
8. Invalid / non-CSV upload rejection.
9. Empty CSV handling.
10. Evaluation metrics API (/api/metrics).
11. Health check endpoint (/api/health).
12. Home dashboard route rendering (/ endpoint).
13. Sample CSV download endpoint (/api/download-sample-csv).
14. Malformed JSON / invalid request payload handling.
15. Batch DataFrame calculation consistency (percentages & averages).
16. Model checkpoint files integrity on disk.
"""

import os
import sys
import unittest
import io
import json

try:
    import pandas as pd
except ImportError:
    pd = None

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from app import app
from inference import SentimentClassifier


class TestDistilBERTReviewIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize test client and load fine-tuned model once."""
        cls.client = app.test_client()
        cls.client.testing = True
        cls.classifier = SentimentClassifier()

    def test_01_model_loading(self):
        """Verify model weights and tokenizer or calibrated engine loads correctly."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(len(self.classifier.id2label), 2)
        self.assertTrue(hasattr(self.classifier, 'device'))

    def test_02_positive_review_prediction(self):
        """Test inference on an unambiguously positive review."""
        review = "This film is an absolute masterpiece with stellar performances and brilliant directing!"
        res = self.classifier.predict(review)

        self.assertEqual(res['prediction'], 'Positive')
        self.assertGreaterEqual(res['confidence'], 0.70)
        self.assertIn('confidence_percentage', res)
        self.assertGreater(res['positive_prob'], res['negative_prob'])
        self.assertGreater(res['token_count'], 0)

    def test_03_negative_review_prediction(self):
        """Test inference on an unambiguously negative review."""
        review = "A catastrophic disaster. Terrible acting, dreadful script, and utterly boring."
        res = self.classifier.predict(review)

        self.assertEqual(res['prediction'], 'Negative')
        self.assertGreaterEqual(res['confidence'], 0.70)
        self.assertGreater(res['negative_prob'], res['positive_prob'])

    def test_04_empty_input_validation(self):
        """Verify that empty or whitespace strings raise ValueError in inference and 400 in API."""
        with self.assertRaises(ValueError):
            self.classifier.predict("")

        with self.assertRaises(ValueError):
            self.classifier.predict("    ")

        # Test via API
        response = self.client.post(
            '/api/predict',
            data=json.dumps({'text': '   '}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data['status'], 'error')

    def test_05_long_review_handling(self):
        """Verify that long reviews exceeding typical context are safely truncated without error."""
        long_review = "This was an interesting movie. " * 300  # ~1200 words
        res = self.classifier.predict(long_review, max_length=128)
        self.assertIn(res['prediction'], ['Positive', 'Negative'])
        self.assertLessEqual(res['token_count'], 128)

    def test_06_batch_inference_list(self):
        """Test vectorized batch inference on a list of reviews."""
        reviews = [
            "An astonishing, emotionally devastating work of modern genius.",
            "Complete unmitigated garbage from start to finish.",
            "Decent acting though the plot leaves something to be desired."
        ]
        results = self.classifier.predict_batch(reviews)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['predicted_sentiment'], 'POSITIVE')
        self.assertEqual(results[1]['predicted_sentiment'], 'NEGATIVE')
        for r in results:
            self.assertIn('confidence', r)
            self.assertIn('probabilities', r)

    def test_07_csv_batch_upload_valid(self):
        """Test /api/batch-predict with a valid multi-row CSV file."""
        csv_content = (
            "id,text\n"
            "1,\"Visionary direction and great performances.\"\n"
            "2,\"A dreadful, painful waste of two hours.\"\n"
            "3,\"A hilarious comedy that had everyone laughing.\"\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_reviews.csv')
        }
        response = self.client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

        res_data = response.get_json()
        self.assertEqual(res_data['status'], 'success')
        self.assertEqual(res_data['data']['total_reviews'], 3)
        self.assertGreaterEqual(res_data['data']['positive_count'], 1)
        self.assertGreaterEqual(res_data['data']['negative_count'], 1)
        self.assertEqual(len(res_data['data']['records']), 3)

    def test_08_invalid_file_upload(self):
        """Test rejection of non-CSV file uploads."""
        data = {
            'file': (io.BytesIO(b"dummy image content"), 'image.png')
        }
        response = self.client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = response.get_json()
        self.assertEqual(res_data['status'], 'error')

    def test_09_empty_csv_upload(self):
        """Test rejection of empty CSV files."""
        data = {
            'file': (io.BytesIO(b""), 'empty.csv')
        }
        response = self.client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_10_evaluation_metrics_api(self):
        """Verify that /api/metrics loads real disk results."""
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('metrics', data)
        self.assertIn('training_history', data)

        # Verify key baseline vs fine-tuned numbers if metrics file exists
        if 'fine_tuned' in data['metrics']:
            ft = data['metrics']['fine_tuned']
            self.assertAlmostEqual(ft['accuracy'], 0.8410, places=2)
            self.assertAlmostEqual(ft['f1_score'], 0.8349, places=2)

    def test_11_health_check_endpoint(self):
        """Verify /api/health reports system status."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])

    def test_12_dashboard_html_rendering(self):
        """Verify home dashboard renders HTTP 200 and loads key UI sections."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('DistilBERT Review Intelligence', html)
        self.assertIn('Neural Inference', html)
        self.assertIn('Batch CSV Pipeline', html)
        self.assertIn('Mahesh Kadam', html)
        self.assertIn('202401120074', html)

    def test_13_download_sample_csv(self):
        """Test /api/download-sample-csv returns valid downloadable CSV."""
        response = self.client.get('/api/download-sample-csv')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/csv')
        content = response.get_data(as_text=True)
        self.assertIn('text', content)

    def test_14_malformed_json_request(self):
        """Test sending empty or malformed body to /api/predict returns 400."""
        response = self.client.post(
            '/api/predict',
            data='not valid json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_15_batch_dashboard_calculations(self):
        """Verify math consistency of batch DataFrame analytics."""
        sample_records = [
            {'text': 'Brilliant masterpiece of storytelling.'},
            {'text': 'Horrible waste of time.'},
            {'text': 'Loved every minute of it.'},
            {'text': 'Disgusting and boring.'}
        ]
        if pd is not None:
            df = pd.DataFrame(sample_records)
        else:
            df = sample_records

        analysis = self.classifier.analyze_dataframe(df)
        self.assertEqual(analysis['total_reviews'], 4)
        self.assertEqual(analysis['positive_count'] + analysis['negative_count'], 4)
        self.assertAlmostEqual(analysis['positive_percentage'] + analysis['negative_percentage'], 100.0, places=1)
        self.assertGreaterEqual(analysis['average_confidence'], 0.5)

    def test_16_model_checkpoint_files_exist(self):
        """Verify presence and non-zero size of serialized model config files."""
        model_dir = os.path.join(PROJECT_DIR, 'models', 'finetuned_distilbert')
        required_files = ['config.json', 'tokenizer.json', 'README_WEIGHTS.md']
        for fname in required_files:
            fpath = os.path.join(model_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing required file: {fname}")
            self.assertGreater(os.path.getsize(fpath), 0, f"File is empty: {fname}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
