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

    def test_model_performance_page(self):
        response = self.client.get("/model-performance")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Performance on unseen test data", response.data)
        self.assertIn(b"Algorithm comparison", response.data)

if __name__ == "__main__": unittest.main()
