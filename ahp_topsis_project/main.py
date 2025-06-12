from ahp import calculate_ahp_weights
from topsis import calculate_topsis_ranking
import pandas as pd

# Đọc file Excel đầu vào
excel_file = pd.ExcelFile("AHP_Van_Chuyen.xlsx")
criteria_df = excel_file.parse("Ma trận tiêu chí")

# Bước 1–3: Tính trọng số tiêu chí bằng AHP
criteria_names, ahp_matrix = calculate_ahp_weights(criteria_df)

# Bước 4–5: Xếp hạng phương án bằng TOPSIS
# (Cần thêm bảng điểm phương án theo từng tiêu chí nếu có)
# dummy_data là bảng ví dụ, thay bằng bảng thực tế nếu bạn có
dummy_data = pd.DataFrame({
    'Phương án': ['Đường bộ', 'Đường biển', 'Hàng không', 'Đường sắt'],
    'Chi phí': [7, 5, 3, 6],
    'Thời gian': [5, 6, 9, 7],
    'Ổn định': [8, 7, 6, 9],
    'An toàn': [6, 8, 7, 6],
    'Linh hoạt': [7, 5, 9, 6]
})

ranking_result = calculate_topsis_ranking(dummy_data, criteria_names, ahp_matrix)

# Hiển thị kết quả
print("\nKết quả xếp hạng TOPSIS:")
print(ranking_result)
