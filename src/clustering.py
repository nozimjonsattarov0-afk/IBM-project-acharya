import pandas as pd
from sklearn.cluster import KMeans
import pickle
import os


def run_clustering():
    df = pd.read_csv('data/processed/cleaned_data.csv')
    X = df[['Age', 'Purchase_Amount']]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    os.makedirs('models', exist_ok=True)
    with open('models/clustering_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    df.to_csv('data/processed/customer_segments.csv', index=False)
    print("Success: Clustering complete.")


if __name__ == "__main__":
    run_clustering()