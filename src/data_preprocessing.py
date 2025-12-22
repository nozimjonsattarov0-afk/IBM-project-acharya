import pandas as pd
import os


def preprocess_data():
    input_path = 'data/raw/customer_purchases.csv'
    if not os.path.exists(input_path):
        print("Error: Raw data not found!")
        return

    df = pd.read_csv(input_path)
    if 'Gender' in df.columns:
        gender_map = {'Erkak': 1, 'Ayol': 0, 'Male': 1, 'Female': 0}
        df['Gender'] = df['Gender'].map(gender_map)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    print("Success: Data preprocessed.")


if __name__ == "__main__":
    preprocess_data()