import unittest
import pandas as pd
from src.currency import inr_to_usd
from src.features import engineer_features

class FeatureTests(unittest.TestCase):
    def test_feature_engineering(self):
        raw = pd.DataFrame([{"Company":"Dell","TypeName":"Notebook","Ram":"8GB","Weight":"1.8kg","ScreenResolution":"IPS Panel 1920x1080","Inches":15.6,"Cpu":"Intel Core i5 7200U 2.5GHz","Memory":"256GB SSD + 1TB HDD","Gpu":"Nvidia GeForce 940MX","OpSys":"Windows 10"}])
        row = engineer_features(raw).iloc[0]
        self.assertEqual(row["Cpu_family"], "Intel Core i5")
        self.assertEqual(row["SSD"], 256); self.assertEqual(row["HDD"], 1024)
        self.assertEqual(row["IPS"], 1); self.assertEqual(row["Gpu_dedicated"], 1)

    def test_currency_conversion(self): self.assertEqual(inr_to_usd(87000, 87), 1000)

if __name__ == "__main__": unittest.main()
