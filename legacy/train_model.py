"""
Training script for the Laptop Price Prediction model.
Run this once from the project root to generate model.pkl and preprocessor.pkl.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv('data/laptop_data.csv')

# ── 2. Feature engineering ──────────────────────────────────────────────────

# Ram: "8GB" → 8
df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)

# Weight: "1.37kg" → 1.37
df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)

# Touchscreen
df['Touchscreen'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'Touchscreen' in str(x) else 0
)

# IPS Panel
df['IPS'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'IPS' in str(x) else 0
)

# PPI (pixels per inch)
def extract_resolution(res_str):
    match = re.search(r'(\d+)x(\d+)', str(res_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1920, 1080  # default

df['X_res'], df['Y_res'] = zip(*df['ScreenResolution'].apply(extract_resolution))
df['PPI'] = ((df['X_res'] ** 2 + df['Y_res'] ** 2) ** 0.5 / df['Inches']).astype(int)

# CPU brand: first word of Cpu
df['Cpu_brand'] = df['Cpu'].apply(lambda x: str(x).split()[0])

# HDD / SSD (GB)
def extract_storage(mem_str, storage_type):
    total = 0
    for part in str(mem_str).split('+'):
        part = part.strip()
        match_tb = re.search(r'(\d+(\.\d+)?)TB', part)
        match_gb = re.search(r'(\d+)GB', part)
        if storage_type.upper() in part.upper():
            if match_tb:
                total += int(float(match_tb.group(1)) * 1024)
            elif match_gb:
                total += int(match_gb.group(1))
    return total

df['HDD'] = df['Memory'].apply(lambda x: extract_storage(x, 'HDD'))
df['SSD'] = df['Memory'].apply(lambda x: extract_storage(x, 'SSD'))

# GPU brand: first word of Gpu
df['Gpu_brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])

# OS category
def simplify_os(os_str):
    os_str = str(os_str)
    if 'Windows' in os_str:
        return 'Windows'
    elif 'macOS' in os_str or 'Mac' in os_str:
        return 'macOS'
    elif 'Linux' in os_str:
        return 'Linux'
    elif 'Chrome' in os_str:
        return 'Chrome OS'
    else:
        return 'Other'

df['OS'] = df['OpSys'].apply(simplify_os)

# Log-transform the target
df['log_Price'] = np.log(df['Price'])

# ── 3. Select features ───────────────────────────────────────────────────────
feature_cols = ['Company', 'TypeName', 'Ram', 'Weight',
                'Touchscreen', 'IPS', 'PPI',
                'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'OS']

X = df[feature_cols]
y = df['log_Price']

cat_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OS']
num_cols = ['Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'HDD', 'SSD']

# ── 4. Build preprocessor ────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', 'passthrough', num_cols)
])

# ── 5. Build & train pipeline ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MAE  (log scale): {mae:.4f}")
print(f"MSE  (log scale): {mse:.4f}")
print(f"R²             : {r2:.4f}")

# Price-scale errors
price_actual = np.exp(y_test)
price_pred   = np.exp(y_pred)
price_mae    = mean_absolute_error(price_actual, price_pred)
print(f"\nMAE  (price scale, ₹): {price_mae:,.0f}")

# ── 7. Save artifacts ────────────────────────────────────────────────────────
# Save the full pipeline as the "model" (it contains the preprocessor inside)
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save the preprocessor separately (for utils.py to use)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Save model performance metrics
model_metrics = {
    'mae_log': mae,
    'mse_log': mse,
    'r2_score': r2,
    'mae_price': price_mae,
    'y_test': y_test.tolist(),
    'y_pred': y_pred.tolist()
}
with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)

# Save unique values for dropdown menus
dropdown_data = {
    'companies': sorted(df['Company'].unique().tolist()),
    'types':     sorted(df['TypeName'].unique().tolist()),
    'cpus':      sorted(df['Cpu_brand'].unique().tolist()),
    'gpus':      sorted(df['Gpu_brand'].unique().tolist()),
    'oses':      sorted(df['OS'].unique().tolist()),
}
with open('dropdown_data.pkl', 'wb') as f:
    pickle.dump(dropdown_data, f)

print("\n✅  Saved model.pkl, preprocessor.pkl, dropdown_data.pkl, model_metrics.pkl")
print("    You can now run: python -m flask --app src/main.py run")