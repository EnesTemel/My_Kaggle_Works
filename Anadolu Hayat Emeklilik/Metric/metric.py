import pandas as pd
from typing import Dict, Optional

def determine_coeffs(filename: str) -> Dict[str, float]:
    df = pd.read_csv(filename)
    label_counts = df.groupby('LABEL').size()
    inverse_freq: Dict[str, float] = {k: 1 / v for k, v in label_counts.items()}
    normalized_coeffs: Dict[str, float] = {k: v / sum(inverse_freq.values()) for k, v in inverse_freq.items()}
    return normalized_coeffs

if __name__ == "__main__":
    coeffs = determine_coeffs(r'C:\Users\enesm\OneDrive\Masaüstü\KAGGLE\Anadolu Hayat Emeklilik\Metric\train.csv')
    if coeffs:
        print("Normalized Coefficients:")
        for label, coeff in coeffs.items():
            print(f"{label}: {coeff}")
