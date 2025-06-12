import pandas as pd
import numpy as np

def calculate_topsis_ranking(df, criteria, weights):
    X = df[criteria].values.astype(float)
    norm = np.linalg.norm(X, axis=0)
    X_norm = X / norm

    weighted = X_norm * weights

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

    scores = dist_worst / (dist_best + dist_worst)

    df_result = df.copy()
    df_result["Điểm TOPSIS"] = scores
    df_result["Xếp hạng"] = scores.argsort()[::-1] + 1
    return df_result.sort_values("Xếp hạng")
