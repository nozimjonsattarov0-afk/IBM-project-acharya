import pandas as pd
import os

def apply_feature_engineering():
    input_path = 'data/processed/cleaned_data.csv'
    if not os.path.exists(input_path):
        print("Error: cleaned_data.csv not found!")
        return

    df = pd.read_csv(input_path)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 45, 100], labels=[0, 1, 2])

    output_path = 'data/processed/featured_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Success: Features engineered and saved to {output_path}")

if __name__ == "__main__":
    apply_feature_engineering()