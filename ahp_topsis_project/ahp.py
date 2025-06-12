import numpy as np
import pandas as pd
import streamlit as st

def calculate_ahp_weights(df):
    try:
        ahp_matrix = df.iloc[3:8, 1:6]
        ahp_matrix = ahp_matrix.applymap(lambda x: eval(str(x)) if isinstance(x, str) else x)
        ahp_matrix = ahp_matrix.astype(float)
    except Exception as e:
        raise ValueError(f"Không thể chuyển dữ liệu thành số: {e}")

    if ahp_matrix.isnull().values.any():
        raise ValueError("Ma trận AHP có giá trị rỗng hoặc không hợp lệ.")

    eigvals, eigvecs = np.linalg.eig(ahp_matrix.values)
    max_index = np.argmax(eigvals.real)
    weights = eigvecs[:, max_index].real
    weights = weights / weights.sum()

    criteria = ['Chi phí', 'Thời gian', 'Ổn định', 'An toàn', 'Linh hoạt']
    return criteria, weights

