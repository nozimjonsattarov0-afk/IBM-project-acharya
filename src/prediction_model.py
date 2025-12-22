import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def fill_prediction_model():
    data_path = 'data/processed/customer_segments.csv'
    if not os.path.exists(data_path):
        print("Error: customer_segments.csv not found. Run clustering.py first!")
        return

    df = pd.read_csv(data_path)

    X = df[['Age', 'Gender', 'Cluster']]
    y = df['Purchase_Amount']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    model_path = 'models/prediction_model.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Success: {model_path} has been updated and filled.")

if __name__ == "__main__":
    fill_prediction_model()