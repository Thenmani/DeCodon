from core.ml_predictor import DropletFreezingPredictor
from core.gemini_agent import generate_mechanistic_report

if __name__ == "__main__":
    predictor = DropletFreezingPredictor()
    result = predictor.analyze_mutation("TDP43", "G298S")

    print("\n=== 1. DeCodon Physics & ML Output ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== 2. DeCodon Gemini Report ===")
    print(generate_mechanistic_report(result))