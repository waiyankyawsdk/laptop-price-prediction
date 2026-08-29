import unittest
from src.main import create_app

def sample_payload():
    return {"Company":"Dell","TypeName":"Notebook","Ram":8,"Weight":1.8,"Touchscreen":0,"IPS":1,"PPI":141,"Cpu_family":"Intel Core i5","Cpu_speed_ghz":2.5,"HDD":0,"SSD":512,"Flash":0,"Hybrid":0,"Gpu_vendor":"Intel","Gpu_dedicated":0,"OS":"Windows"}

class ApiTests(unittest.TestCase):
    def setUp(self): self.client = create_app({"TESTING": True}).test_client()

    def test_prediction_api(self):
        response = self.client.post("/api/predict", json=sample_payload())
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreater(body["predicted_price_inr"], 0); self.assertGreater(body["predicted_price_usd"], 0)

    def test_missing_field(self):
        payload = sample_payload(); payload.pop("Ram")
        self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 400)

    def test_prediction_page_preserves_submitted_values(self):
        payload = sample_payload()
        payload.update({"Company": "Apple", "Ram": 16, "Touchscreen": 1, "IPS": 0})
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('value="Apple" selected', html)
        self.assertIn('name="Ram" value="16"', html)
        self.assertIn('name="Touchscreen"', html)
        self.assertIn('value="1" selected', html)
        self.assertIn("Estimated fair price", html)

    def test_metrics_html_page(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Research results", html)
        self.assertIn("RandomForest", html)
        self.assertIn("Mean absolute error", html)

    def test_metrics_api_remains_json(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        self.assertIn("test_metrics", response.get_json())

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["model"], "RandomForest")

if __name__ == "__main__": unittest.main()
