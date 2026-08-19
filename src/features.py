"""Feature engineering shared by training and inference."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

MODEL_FEATURES = ["Company", "TypeName", "Ram", "Weight", "Touchscreen", "IPS", "PPI", "Cpu_family", "Cpu_speed_ghz", "HDD", "SSD", "Flash", "Hybrid", "Gpu_vendor", "Gpu_dedicated", "OS"]
CATEGORICAL_FEATURES = ["Company", "TypeName", "Cpu_family", "Gpu_vendor", "OS"]
NUMERIC_FEATURES = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]

def _number(pattern, value, default=0.0):
    match = re.search(pattern, str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else default

def cpu_family(cpu):
    text = str(cpu)
    patterns = [(r"Intel Core i9", "Intel Core i9"), (r"Intel Core i7", "Intel Core i7"), (r"Intel Core i5", "Intel Core i5"), (r"Intel Core i3", "Intel Core i3"), (r"Intel Core M", "Intel Core M"), (r"Intel Celeron", "Intel Celeron"), (r"Intel Pentium", "Intel Pentium"), (r"AMD Ryzen", "AMD Ryzen"), (r"AMD [A-Z]-Series|AMD A[0-9]", "AMD A-Series"), (r"AMD E-Series", "AMD E-Series"), (r"Apple M[0-9]", "Apple Silicon")]
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE): return label
    if text.startswith("Intel"): return "Intel Other"
    if text.startswith("AMD"): return "AMD Other"
    return text.split()[0] if text.strip() else "Other"

def os_family(value):
    text = str(value).lower()
    if "windows" in text: return "Windows"
    if "mac" in text: return "macOS"
    if "linux" in text: return "Linux"
    if "chrome" in text: return "Chrome OS"
    if "android" in text: return "Android"
    return "Other"

def storage_capacity(memory, storage_type):
    total = 0
    for part in str(memory).split("+"):
        if storage_type.lower() not in part.lower(): continue
        value = _number(r"([0-9.]+)\s*(?:TB|GB)", part)
        if "TB" in part.upper(): value *= 1024
        total += int(value)
    return total

def engineer_features(raw):
    df = raw.copy(); result = pd.DataFrame(index=df.index)
    result["Company"] = df["Company"].astype(str).str.strip()
    result["TypeName"] = df["TypeName"].astype(str).str.strip()
    result["Ram"] = df["Ram"].astype(str).str.extract(r"([0-9]+)")[0].astype(float)
    result["Weight"] = df["Weight"].astype(str).str.extract(r"([0-9.]+)")[0].astype(float)
    screen = df["ScreenResolution"].astype(str)
    result["Touchscreen"] = screen.str.contains("Touchscreen", case=False).astype(int)
    result["IPS"] = screen.str.contains("IPS", case=False).astype(int)
    resolution = screen.str.extract(r"([0-9]+)x([0-9]+)").astype(float)
    x_res, y_res = resolution[0].fillna(1920), resolution[1].fillna(1080)
    inches = pd.to_numeric(df["Inches"], errors="coerce").fillna(15.6)
    result["PPI"] = np.sqrt(x_res**2 + y_res**2) / inches
    result["Cpu_family"] = df["Cpu"].map(cpu_family)
    result["Cpu_speed_ghz"] = df["Cpu"].map(lambda v: _number(r"([0-9.]+)\s*GHz", v))
    for feature, label in [("HDD", "HDD"), ("SSD", "SSD"), ("Flash", "Flash Storage"), ("Hybrid", "Hybrid")]: result[feature] = df["Memory"].map(lambda v, label=label: storage_capacity(v, label))
    gpu = df["Gpu"].astype(str)
    result["Gpu_vendor"] = gpu.str.split().str[0].replace({"ATI": "AMD"})
    result["Gpu_dedicated"] = gpu.str.contains(r"NVIDIA|GeForce|Radeon RX|Radeon Pro", case=False, regex=True).astype(int)
    result["OS"] = df["OpSys"].map(os_family)
    return result[MODEL_FEATURES]

def prediction_frame(payload):
    missing = [name for name in MODEL_FEATURES if name not in payload]
    if missing: raise ValueError(f"Missing fields: {', '.join(missing)}")
    frame = pd.DataFrame([{name: payload[name] for name in MODEL_FEATURES}])
    for column in NUMERIC_FEATURES: frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame

