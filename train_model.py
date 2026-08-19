"""Train and evaluate the laptop price prediction system."""
from pathlib import Path
from src.training import train_and_evaluate

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = train_and_evaluate(root / "data/laptop_data.csv", root / "artifacts", root / "reports")
    print("\nTraining completed")
    print(f"Selected model: {result['selected_model']}")
    print(f"Test R²: {result['test_metrics']['r2']:.4f}")
    print(f"Test MAE: INR {result['test_metrics']['mae_inr']:,.0f}")
    print(f"Test RMSE: INR {result['test_metrics']['rmse_inr']:,.0f}")

