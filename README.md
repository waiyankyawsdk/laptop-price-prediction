# Laptop Price Prediction Research Baseline

Complete reproducible implementation for the laptop-price prediction papers. It trains on the included 1,303-row historical dataset, evaluates multiple regressors, predicts native **Indian rupee (INR)** prices, optionally displays converted **US dollar (USD)** values, and provides a Flask UI and JSON API.

## 1. Understand the problem

This is regression: `laptop specifications -> estimated INR price`.

- **R²** is explained variation, not percentage accuracy.
- **MAE** is average absolute error in the stated unit.
- **RMSE** penalizes large mistakes more strongly.
- **MAPE** is average relative error.
- Training uses `log(price)` to reduce the influence of premium outliers; user-facing metrics are converted back to INR.

## 2. Currency

The included `Price` target is treated as INR. USD is calculated as `predicted INR / INR_PER_USD`; it is not a separately trained US-market price. The default is 87 INR per USD. Override and record the rate used:

```bash
INR_PER_USD=88.5 python run.py
```

## 3. Structure

```text
data/laptop_data.csv   historical dataset
src/features.py        shared feature engineering
src/training.py        CV, calibration and test evaluation
src/currency.py        INR-to-USD conversion
src/main.py            Flask UI and API
tests/                 automated tests
artifacts/             generated model and metadata
reports/               generated evaluation results
legacy/                original supplied code for comparison
```

## 4. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows activation: `.venv\Scripts\activate`

## 5. Train and evaluate

```bash
python train_model.py
```

The program reads and engineers the data, creates 70% training / 15% calibration / 15% untouched test partitions, compares Ridge, Gradient Boosting, Random Forest and Extra Trees on identical five-fold CV partitions, selects the lowest CV log-RMSE model, builds a 90% conformal interval, and evaluates once on the test data.

Generated outputs:

```text
artifacts/model.joblib
artifacts/options.joblib
artifacts/metadata.json
reports/model_comparison.csv
reports/test_predictions.csv
reports/evaluation_summary.json
```

Use these generated values in your paper rather than copying another paper's results.

## 6. Test and run

```bash
python -m unittest discover -s tests -v
python run.py
```

Open <http://127.0.0.1:5000>.

- `/predict` - prediction form
- `/methodology` - evaluation explanation
- `/api/predict` - JSON API
- `/api/metrics` - complete metrics

API example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d @examples/sample_request.json
```

## 7. Relationship to reference papers

The papers commonly reuse the approximately 1,303-row dataset and report ensemble R² values around 0.89-0.93. Exact replication is not expected unless split, seed, features, transformation and hyperparameters are identical. This project avoids hard-coded results and uses a fixed seed, shared preprocessing, identical CV folds, separate calibration/test sets, INR-scale metrics, and explicit uncertainty/currency notes.

## 8. Limitation and next phase

This old static dataset is a university baseline, not a current retail model. For your contribution, collect dated multi-retailer records including seller, country, currency, exchange rate, discount, exact CPU/GPU generation, NPU, RAM type, SSD generation, display, battery, warranty and condition. Evaluate random, SKU-grouped, future-period, retailer-held-out and country-held-out tests.

Suggested title: **Explainable and Uncertainty-Aware Laptop Price Prediction Using Machine Learning**.
