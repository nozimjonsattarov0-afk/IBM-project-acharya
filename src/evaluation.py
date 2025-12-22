import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


def evaluate_model():
    df = pd.read_csv('data/processed/customer_segments.csv')

    with open('models/prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    X = df[['Age', 'Gender', 'Cluster']]
    y_true = df['Purchase_Amount']
    y_pred = model.predict(X)

    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.2f}")


if __name__ == "__main__":
    evaluate_model()