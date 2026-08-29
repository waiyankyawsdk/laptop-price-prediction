# Step-by-Step Implementation Guide

## Step 1 - Establish the baseline

Your baseline question is: **Can historical laptop specifications estimate a laptop's price in Indian rupees?**

Input features include brand, category, RAM, weight, display, CPU, storage, GPU and OS. The target is `Price` in INR. This is supervised regression because the target is continuous.

The included data has 1,303 historical records. It is useful for reproducing the reference-paper workflow, but it is not a current 2026 market dataset.

## Step 2 - Understand feature engineering

Open `src/features.py`. `engineer_features()` converts raw strings into consistent features:

- `8GB` -> RAM `8`
- `1.80kg` -> weight `1.80`
- `1920x1080` and screen inches -> PPI
- screen text -> touchscreen and IPS flags
- CPU text -> CPU family and clock speed
- `256GB SSD + 1TB HDD` -> SSD `256`, HDD `1024`
- GPU text -> vendor and dedicated-GPU flag
- detailed operating system -> OS family

The same code is used during training and prediction. This prevents training-serving mismatch.

## Step 3 - Understand the log target

Laptop prices are right-skewed: a small number of premium laptops are much more expensive. Training uses:

```python
y_log = np.log(price_inr)
```

Prediction returns to the original unit:

```python
price_inr = np.exp(predicted_log_price)
```

Therefore, `mae_log` is not a rupee error. Use `mae_inr` when explaining business meaning.

## Step 4 - Understand the data split

The implementation uses:

- 70% training: fit and compare algorithms
- 15% calibration: calculate uncertainty interval width
- 15% test: final evaluation only

Five-fold cross-validation occurs only in the training partition. The best mean CV log-RMSE model is selected.

## Step 5 - Understand the algorithms

- **Ridge** is the interpretable linear baseline.
- **Gradient Boosting** builds trees sequentially to correct earlier errors.
- **Random Forest** averages many bootstrapped decision trees.
- **Extra Trees** uses more randomized tree construction.

The reproduced run selected Random Forest. This means it performed best under this implementation's features, partitions and scoring—not that it is universally best.

## Step 6 - Read the reproduced evaluation

The verified run produced:

| Measure | Result |
|---|---:|
| Selected model | Random Forest |
| Five-fold CV R² | 0.8741 ± 0.0129 |
| Test R² | 0.8740 |
| Test MAE | INR 9,723 |
| Test RMSE | INR 15,691 |
| Test median absolute error | INR 5,419 |
| Test MAPE | 17.09% |
| Nominal interval | 90% |
| Empirical test coverage | 91.84% |

Interpretation: the model explains about 87.4% of log-price variation on the held-out records. Its average absolute original-scale error is about INR 9,723. The higher RMSE shows that some mistakes are considerably larger. These results are not directly comparable with a paper unless its split, target scale and feature pipeline are identical.

## Step 7 - Train locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

Inspect:

- `reports/model_comparison.csv`
- `reports/test_predictions.csv`
- `reports/evaluation_summary.json`
- `artifacts/metadata.json`

## Step 8 - Run automated checks

```bash
python -m unittest discover -s tests -v
```

The tests verify storage/CPU/display feature parsing, INR-to-USD conversion, valid API prediction and missing-field rejection.

## Step 9 - Run the user interface

```bash
python run.py
```

Visit `http://127.0.0.1:5000/predict`, enter specifications and submit. The result includes:

- estimated INR price;
- converted USD display;
- 90% INR prediction interval;
- disclosed INR-per-USD rate.

## Step 10 - Use USD correctly

The model has learned Indian-market historical values, so the USD value is only currency conversion:

```text
USD = INR prediction / INR per USD
```

Set the experiment rate:

```bash
INR_PER_USD=88.5 python run.py
```

For a true US price predictor, collect US retailer observations with USD targets and retrain. Currency conversion cannot learn differences in US taxes, retailer margins, product availability or promotion behavior.

## Step 11 - Write your baseline-results section


> Four regression algorithms were evaluated using a fixed 70/15/15 training, calibration, and testing design. Model selection used five-fold cross-validation within the training partition. Random Forest achieved the lowest mean cross-validation log-RMSE and was selected for final evaluation. On the untouched test partition, it obtained R² = 0.874, MAE = INR 9,723, RMSE = INR 15,691, and MAPE = 17.09%. A split-conformal interval calibrated at the 90% nominal level covered 91.84% of test prices. These results represent within-dataset baseline performance and should not be interpreted as evidence of present-day market accuracy.

## Step 12 - Implement your research contribution

Do not stop at the baseline. Create `data/current_laptop_prices.csv` with a documented collection protocol and fields such as:

- observation date, release date and retailer;
- country, native currency and exchange rate;
- list price, sale price, tax and discount;
- exact brand, series, model and SKU;
- CPU/GPU exact model, generation and benchmark;
- NPU availability/TOPS;
- RAM capacity/type/speed/upgradeability;
- storage capacity/type/interface;
- display resolution/panel/refresh rate/brightness;
- battery Wh, weight, warranty and condition.

Then add grouped and time-aware experiments. The most important comparison is **old static baseline versus current time-aware model**, not simply Random Forest versus another algorithm.

