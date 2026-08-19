"""INR-native prediction with optional USD display conversion."""
import os

def inr_per_usd():
    rate = float(os.getenv("INR_PER_USD", "87.0"))
    if rate <= 0: raise ValueError("INR_PER_USD must be greater than zero")
    return rate

def inr_to_usd(price_inr, rate=None): return float(price_inr) / (rate or inr_per_usd())

