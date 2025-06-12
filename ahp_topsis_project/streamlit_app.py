import streamlit as st
import pandas as pd
import numpy as np
from ahp import calculate_ahp_weights
from topsis import calculate_topsis_ranking
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from database import get_all_criteria, get_all_alternatives, save_evaluation_session, get_evaluation_history, add_criterion, add_alternative
from matplotlib.gridspec import GridSpec
import fractions

# Hàm hiển thị stepper
STEP_LABELS = ["Đăng nhập", "Chọn tiêu chí & phương án", "Nhập ma trận", "Kết quả"]
def show_stepper(current_step):
    step_html = "<div style='display:flex;align-items:center;justify-content:center;margin-bottom:30px;'>"
    for i, label in enumerate(STEP_LABELS):
        color = "#1976d2" if i == current_step else ("#bdbdbd" if i < current_step else "#fff")
        text_color = "#fff" if i == current_step else ("#333" if i < current_step else "#333")
        border = "3px solid #1976d2" if i == current_step else "2px solid #bdbdbd"
        step_html += f"""
        <div style='display:flex;flex-direction:column;align-items:center;'>
            <div style='width:48px;height:48px;border-radius:50%;background:{color};border:{border};display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;color:{text_color};transition:background 0.3s;'>{i+1}</div>
            <div style='margin-top:6px;font-size:14px;color:#333;text-align:center;width:90px;height:36px;display:flex;align-items:center;justify-content:center;line-height:1.2'>{label}</div>
        </div>
        """
        if i < len(STEP_LABELS)-1:
            step_html += "<div style='height:2px;width:48px;background:#bdbdbd;margin:0 6px;'></div>"
    step_html += "</div>"
    st.markdown(step_html, unsafe_allow_html=True)

st.set_page_config(page_title="AHP + TOPSIS", layout="wide")
st.title("🌟 Hệ thống hỗ trợ ra quyết định AHP + TOPSIS")

# Xác định bước hiện tại
current_step = 0
username = st.sidebar.text_input("Tên người dùng", key="sidebar_username")
if username:
    current_step = 1
    # Lấy danh sách tiêu chí và phương án từ database
    criteria_df = get_all_criteria()
    alternatives_df = get_all_alternatives()
    selected_criteria = st.session_state.get('selected_criteria', criteria_df['name'].tolist()[:4])
    selected_alternatives = st.session_state.get('selected_alternatives', alternatives_df['name'].tolist())
    if len(selected_criteria) >= 4 and len(selected_alternatives) >= 2:
        current_step = 2
        if (
            "ahp_matrix" in st.session_state and
            st.session_state.get('show_ahp_details', False)
        ):
            current_step = 3
show_stepper(current_step)

# Bước 1: Đăng nhập hoặc cấu hình
st.sidebar.header("🔐 Bước 1: Đăng nhập")
if username:
    st.sidebar.success(f"Xin chào, {username}!")
    
    # Hiển thị lịch sử đánh giá
    st.sidebar.header("📜 Lịch sử đánh giá")
    history = get_evaluation_history(username)
    if not history.empty:
        history = history.sort_values('created_at', ascending=False).reset_index(drop=True)
        st.sidebar.dataframe(
            history[['created_at', 'num_criteria', 'num_alternatives']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.sidebar.info("Chưa có lịch sử đánh giá")

    # Bước 2: Nhập tiêu chí và phương án
    st.header("📌 Bước 2: Chọn tiêu chí và phương án")
    
    # Lấy danh sách tiêu chí và phương án từ database
    criteria_df = get_all_criteria()
    alternatives_df = get_all_alternatives()
    

    # Hiển thị danh sách tiêu chí có sẵn
    st.subheader("Tiêu chí có sẵn:")
    st.dataframe(criteria_df[['name', 'description']])
    
    # Cho phép thêm tiêu chí mới
    with st.expander("Thêm tiêu chí mới"):
        new_criterion_name = st.text_input("Tên tiêu chí mới")
        new_criterion_desc = st.text_input("Mô tả tiêu chí")
        if st.button("Thêm tiêu chí") and new_criterion_name:
            if add_criterion(new_criterion_name, new_criterion_desc):
                st.success(f"Đã thêm tiêu chí: {new_criterion_name}")
                st.rerun()
            else:
                st.error("Không thể thêm tiêu chí. Có thể tên tiêu chí đã tồn tại.")
    
    # --- Import file Excel và tự động cập nhật vào các bảng ở Bước 3 và 4 ---
    st.subheader("📥 Import dữ liệu từ file Excel")
    uploaded_file = st.file_uploader("Chọn file Excel để import", type=["xlsx", "xls"])
    if uploaded_file is not None and not st.session_state.get("imported", False):
        try:
            excel_data = pd.read_excel(uploaded_file, sheet_name=None, header=None)
            # Ưu tiên sheet 'Ma trận tổng hợp' nếu có
            matrix_sheet = None
            for sheet_name in excel_data:
                if 'ma trận tổng hợp' in sheet_name.lower():
                    matrix_sheet = excel_data[sheet_name]
                    break
            if matrix_sheet is not None:
                from fractions import Fraction
                def float_to_ahp_str(val):
                    ahp_scale = ["1/9", "1/8", "1/7", "1/6", "1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                    ahp_scale_values = [float(Fraction(s)) for s in ahp_scale]
                    diffs = [abs(val - v) for v in ahp_scale_values]
                    idx = diffs.index(min(diffs))
                    return ahp_scale[idx]
                df = matrix_sheet
                i = 0
                first_alt_header = None
                while i < len(df):
                    row = df.iloc[i]
                    if isinstance(row[0], str) and '[Ma trận tiêu chí]' in row[0]:
                        # Đọc ma trận tiêu chí
                        header = df.iloc[i+1, 1:].dropna().tolist()
                        n = len(header)
                        matrix = []
                        for j in range(i+2, i+2+n):
                            matrix.append([float(x) for x in df.iloc[j, 1:1+n]])
                        matrix_df = pd.DataFrame(matrix, index=header, columns=header)
                        st.session_state['selected_criteria'] = header
                        st.session_state.ahp_matrix = matrix_df
                        ahp_matrix_select = {}
                        for r in header:
                            ahp_matrix_select[r] = {c: float_to_ahp_str(matrix_df.loc[r, c]) for c in header}
                        st.session_state.ahp_matrix_select = ahp_matrix_select
                        i = i+2+n
                    elif isinstance(row[0], str) and '[Ma trận phương án' in row[0]:
                        import re
                        m = re.search(r'\[Ma trận phương án (.+)\]', row[0])
                        if m:
                            crit = m.group(1).strip()
                            header = df.iloc[i+1, 1:].dropna().tolist()
                            n = len(header)
                            matrix = []
                            for j in range(i+2, i+2+n):
                                matrix.append([float(x) for x in df.iloc[j, 1:1+n]])
                            matrix_df = pd.DataFrame(matrix, index=header, columns=header)
                            st.session_state[f"ahp_alt_matrix_{crit}"] = matrix_df
                            if not first_alt_header:
                                first_alt_header = header
                                st.session_state['selected_alternatives'] = header
                            i = i+2+n
                        else:
                            i += 1
                    else:
                        i += 1
            st.session_state["imported"] = True  # ✅ đánh dấu đã import xong
            st.success("✅ Đã import dữ liệu thành công. Ma trận đã được cập nhật.")
            st.rerun()  # 🔁 Tải lại app để phản ánh thay đổi
            st.stop()   # 🛑 Dừng vòng lặp hiện tại để tránh lỗi session_state
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file Excel: {e}")

    # Chọn tiêu chí để sử dụng
    selected_criteria = st.multiselect(
        "Chọn các tiêu chí đánh giá:",
        options=criteria_df['name'].tolist(),
        key="selected_criteria"
    )
    
    # Hiển thị cảnh báo nếu chưa đủ điều kiện
    if len(selected_criteria) < 4 or len(selected_alternatives) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 4 tiêu chí và 2 phương án để tiếp tục")

    # Hiển thị danh sách phương án có sẵn
    st.subheader("Phương án có sẵn:")
    st.dataframe(alternatives_df[['name', 'description']])
    
    # Cho phép thêm phương án mới
    with st.expander("Thêm phương án mới"):
        new_alt_name = st.text_input("Tên phương án mới")
        new_alt_desc = st.text_input("Mô tả phương án")
        if st.button("Thêm phương án") and new_alt_name:
            if add_alternative(new_alt_name, new_alt_desc):
                st.success(f"Đã thêm phương án: {new_alt_name}")
                st.rerun()
            else:
                st.error("Không thể thêm phương án. Có thể tên phương án đã tồn tại.")
    
    # Chọn phương án để đánh giá
    selected_alternatives = st.multiselect(
        "Chọn các phương án đánh giá:",
        options=alternatives_df['name'].tolist(),
        key="selected_alternatives"
    )
    # Sau multiselect, luôn lấy lại từ session_state để render các bước tiếp theo
    selected_criteria = st.session_state.get('selected_criteria', criteria_df['name'].tolist()[:4])
    selected_alternatives = st.session_state.get('selected_alternatives', alternatives_df['name'].tolist())

    if len(selected_criteria) >= 4 and len(selected_alternatives) >= 2:

        # Bước 3: Nhập ma trận AHP
        st.header("📊 Bước 3: Nhập ma trận so sánh AHP")

        ahp_scale = ["1/9", "1/8", "1/7", "1/6", "1/5", "1/4", "1/3", "1/2", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        ahp_scale_values = {s: float(fractions.Fraction(s)) for s in ahp_scale}
        if "ahp_matrix_select" not in st.session_state or \
            list(st.session_state.ahp_matrix_select.keys()) != selected_criteria:
            if "ahp_matrix" in st.session_state and list(st.session_state.ahp_matrix.index) == selected_criteria:
                # Nếu đã có ma trận import, tự động tạo ahp_matrix_select từ ma trận này
                matrix_df = st.session_state.ahp_matrix
                ahp_matrix_select = {}
                for i, row in enumerate(selected_criteria):
                    ahp_matrix_select[row] = {}
                    for j, col in enumerate(selected_criteria):
                        val = matrix_df.loc[row, col]
                        ahp_matrix_select[row][col] = str(val)
                st.session_state.ahp_matrix_select = ahp_matrix_select
            else:
                st.session_state.ahp_matrix_select = {}
                for i, row in enumerate(selected_criteria):
                    st.session_state.ahp_matrix_select[row] = {}
                    for j, col in enumerate(selected_criteria):
                        if i == j:
                            st.session_state.ahp_matrix_select[row][col] = "1"
                        else:
                            st.session_state.ahp_matrix_select[row][col] = ""
        matrix = st.session_state.ahp_matrix_select
        updated = False
        # Hiển thị bảng với tiêu đề hàng/cột
        st.write("")
        col_titles = [""] + list(selected_criteria)
        cols = st.columns(len(col_titles))
        for idx, title in enumerate(col_titles):
            with cols[idx]:
                st.markdown(f"<div style='display:flex;align-items:center;justify-content:center;height:40px;font-weight:bold;font-size:18px'>{title}</div>", unsafe_allow_html=True)
        for i, row in enumerate(selected_criteria):
            cols = st.columns(len(selected_criteria) + 1)
            with cols[0]:
                st.markdown(f"<div style='text-align:center;font-weight:bold;margin-top:27px;'>{row}</div>", unsafe_allow_html=True)
            for j, col in enumerate(selected_criteria):
                with cols[j+1]:
                    if i == j:
                        st.markdown("""
                        <div style='width:100%; height:48px; min-height:48px; padding:0 12px;margin-top:19px; font-size:16px; display:flex; align-items:center; justify-content:center; box-sizing:border-box; font-weight:bold;'>1</div>
                        """, unsafe_allow_html=True)
                    elif j < i:
                        val = matrix[selected_criteria[j]][row]
                        if val:
                            inv_val = str(1 / fractions.Fraction(val))
                            st.markdown(f"<div style='display:flex;align-items:center;justify-content:center;height:40px;color:#888;font-size:18px;margin-top:22px;'>{inv_val}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:40px;color:#888;margin-top:22px;;'>?</div>", unsafe_allow_html=True)
                    else:
                        key = f"ahp_{row}_{col}"
                        current = matrix[row][col] if matrix[row][col] else "1"
                        selected = st.selectbox(
                            label="",
                            options=ahp_scale,
                            index=ahp_scale.index(current) if current in ahp_scale else 8,
                            key=key
                        )
                        if selected != matrix[row][col]:
                            matrix[row][col] = selected
                            matrix[col][row] = str(1 / fractions.Fraction(selected))
                            updated = True
        st.session_state.ahp_matrix_select = matrix
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        if st.button("🔁 Cập nhật ma trận đầy đủ (selectbox)"):
            # Chuyển sang DataFrame số thực để tính toán
            ahp_matrix = np.zeros((len(selected_criteria), len(selected_criteria)), dtype=float)
            for i, row in enumerate(selected_criteria):
                for j, col in enumerate(selected_criteria):
                    val = matrix[row][col]
                    if val:
                        ahp_matrix[i, j] = float(fractions.Fraction(val))
                    else:
                        ahp_matrix[i, j] = 1.0 if i == j else np.nan
            st.session_state.ahp_matrix = pd.DataFrame(ahp_matrix, index=selected_criteria, columns=selected_criteria)
            st.session_state.show_ahp_details = True
            st.rerun()

        # Chỉ hiển thị các bảng trung gian nếu đã bấm nút cập nhật
        if st.session_state.get('show_ahp_details', False):
            try:
                # Chuyển đổi các giá trị phân số sang số thực
                def parse_value(val):
                    if isinstance(val, str):
                        val = val.replace(',', '.')
                        try:
                            if '/' in val:
                                return float(fractions.Fraction(val))
                            else:
                                return float(val)
                        except:
                            return np.nan
                    return val
                ahp_matrix = st.session_state.ahp_matrix.applymap(parse_value).values.astype(float)
                n = len(selected_criteria)
                # 1. Tính tổng từng cột
                col_sum = ahp_matrix.sum(axis=0)
                sum_df = pd.DataFrame([col_sum], columns=selected_criteria, index=["Sum"])
                st.markdown("**Tổng từng cột (Sum):**")
                st.dataframe(sum_df)

                # 2. Ma trận chuẩn hóa
                norm_matrix = ahp_matrix / col_sum
                norm_df = pd.DataFrame(norm_matrix, columns=selected_criteria, index=selected_criteria)
                st.markdown("**Ma trận chuẩn hóa:**")
                st.dataframe(norm_df)

                # 3. Trọng số tiêu chí (trung bình từng hàng)
                weights = norm_matrix.mean(axis=1)
                weights_df = pd.DataFrame({"Tiêu chí": selected_criteria, "Trọng số (trung bình hàng)": weights})
                st.markdown("**Trọng số tiêu chí (trung bình từng hàng):**")
                st.dataframe(weights_df)

               
                # 4. Bảng kiểm tra nhất quán
                weighted_sum = np.dot(ahp_matrix, weights)
                consistency_vector = weighted_sum / weights
                ahp_detail_df = pd.DataFrame({
                    'Tiêu chí': selected_criteria,
                    'Weighted sum': weighted_sum,
                    'Trọng số': weights,
                    'Consistency vector': consistency_vector
                })
                st.markdown('**Bảng kiểm tra nhất quán:**')
                st.dataframe(ahp_detail_df)

                # 5. Lambda_max, CI, CR
                lambda_max = consistency_vector.mean()
                ci = (lambda_max - n) / (n - 1)
                ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
                ri = ri_table.get(n, 1.49)
                cr = ci / ri if ri != 0 else 0
                st.markdown(f"""
                **Chỉ số nhất quán:**
                - Lambda_max: `{lambda_max:.4f}`
                - CI (Consistency Index): `{ci:.4f}`
                - CR (Consistency Ratio): `{cr:.4f}`
                """)

                # Hiển thị bảng xếp hạng tiêu chí
                rank_df = weights_df.copy()
                rank_df['Rank'] = rank_df['Trọng số (trung bình hàng)'].rank(ascending=False, method='min').astype(int)
                rank_df = rank_df.sort_values('Rank')
                st.markdown("**Bảng xếp hạng tiêu chí:**")
                st.dataframe(rank_df.rename(columns={"Tiêu chí": "Criteria", "Trọng số (trung bình hàng)": "Criteria Weights"}))


            except Exception as e:
                st.info("Không thể tính toán các bảng trung gian với ma trận hiện tại.")

        # Bước 4: Nhập ma trận so sánh các phương án theo từng tiêu chí
        st.header("📝 Bước 4: Nhập ma trận so sánh các phương án theo từng tiêu chí")
        alt_weights = {}
        alt_crs = {}
        alt_lambdas = {}
        alt_cis = {}
        show_all_valid = True
        for criterion in selected_criteria:
            st.subheader(f"Tiêu chí: {criterion}")
            # SỬA: Chỉ khởi tạo bảng mặc định nếu chưa có dữ liệu import
            if f"ahp_alt_matrix_{criterion}" not in st.session_state or \
                list(st.session_state[f"ahp_alt_matrix_{criterion}"].index) != selected_alternatives or \
                list(st.session_state[f"ahp_alt_matrix_{criterion}"].columns) != selected_alternatives:
                st.session_state[f"ahp_alt_matrix_{criterion}"] = pd.DataFrame(
                    [[1.0 if i == j else np.nan for j in range(len(selected_alternatives))] for i in range(len(selected_alternatives))],
                    index=selected_alternatives, columns=selected_alternatives, dtype=object
                )
            edited_matrix = st.data_editor(st.session_state[f"ahp_alt_matrix_{criterion}"], key=f"ahp_alt_editor_{criterion}", use_container_width=True)
            if st.button(f"Cập nhật ma trận phương án cho tiêu chí '{criterion}'"):
                reflected = edited_matrix.copy()
                for i in range(len(selected_alternatives)):
                    for j in range(i + 1, len(selected_alternatives)):
                        try:
                            val = float(reflected.iat[i, j])
                            reflected.iat[j, i] = round(1 / val, 5) if val != 0 else 0.0
                            reflected.iat[i, i] = 1.0
                            reflected.iat[j, j] = 1.0
                        except:
                            reflected.iat[j, i] = 0.0
                st.session_state[f"ahp_alt_matrix_{criterion}"] = reflected.copy()
                st.session_state[f"show_alt_details_{criterion}"] = True
                st.rerun()
            # Hiển thị các bảng trung gian nếu đã cập nhật
            if st.session_state.get(f"show_alt_details_{criterion}", False):
                try:
                    ahp_matrix = st.session_state[f"ahp_alt_matrix_{criterion}"].values.astype(float)
                    n = len(selected_alternatives)
                    col_sum = ahp_matrix.sum(axis=0)
                    sum_df = pd.DataFrame([col_sum], columns=selected_alternatives, index=["Sum"])
                    st.markdown("**Tổng từng cột (Sum):**")
                    st.dataframe(sum_df)
                    norm_matrix = ahp_matrix / col_sum
                    norm_df = pd.DataFrame(norm_matrix, columns=selected_alternatives, index=selected_alternatives)
                    st.markdown("**Ma trận chuẩn hóa:**")
                    st.dataframe(norm_df)
                    weights = norm_matrix.mean(axis=1)
                    weights_df = pd.DataFrame({"Phương án": selected_alternatives, f"Trọng số PA ({criterion})": weights})
                    st.markdown(f"**Trọng số các phương án theo tiêu chí {criterion} (trung bình từng hàng):**")
                    st.dataframe(weights_df)
                    weighted_sum = np.dot(ahp_matrix, weights)
                    consistency_vector = weighted_sum / weights
                    ahp_detail_df = pd.DataFrame({
                        'Phương án': selected_alternatives,
                        'Weighted sum': weighted_sum,
                        'Trọng số': weights,
                        'Consistency vector': consistency_vector
                    })
                    st.markdown('**Bảng kiểm tra nhất quán:**')
                    st.dataframe(ahp_detail_df)
                    lambda_max = consistency_vector.mean()
                    ci = (lambda_max - n) / (n - 1)
                    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
                    ri = ri_table.get(n, 1.49)
                    cr = ci / ri if ri != 0 else 0
                    st.markdown(f"""
                    **Chỉ số nhất quán:**
                    - Lambda_max: `{lambda_max:.4f}`
                    - CI (Consistency Index): `{ci:.4f}`
                    - CR (Consistency Ratio): `{cr:.4f}`
                    """)
                    alt_weights[criterion] = weights
                    alt_crs[criterion] = cr
                    alt_lambdas[criterion] = lambda_max
                    alt_cis[criterion] = ci
                    if cr > 0.1:
                        show_all_valid = False
                        st.error(f"❌ Ma trận phương án cho tiêu chí '{criterion}' không nhất quán. CR = {cr:.4f} > 0.10")
                except Exception as e:
                    st.info(f"Không thể tính toán các bảng trung gian cho tiêu chí '{criterion}' với ma trận hiện tại.")
        # Sau khi tổng hợp điểm các phương án (AHP), chỉ hiển thị bảng kết quả và xuất file
        if show_all_valid and len(alt_weights) == len(selected_criteria) and len(alt_weights) > 0:
            st.header("📊 Tổng hợp điểm các phương án (AHP)")
            crit_weights = st.session_state.get('ahp_criteria_weights', None)
            if crit_weights is None:
                ahp_matrix = st.session_state.ahp_matrix.values.astype(float)
                norm_matrix = ahp_matrix / ahp_matrix.sum(axis=0)
                crit_weights = norm_matrix.mean(axis=1)
            result = pd.DataFrame({'Phương án': selected_alternatives})
            for i, criterion in enumerate(selected_criteria):
                result[criterion] = alt_weights[criterion]
            result['Điểm tổng hợp'] = 0
            for i, criterion in enumerate(selected_criteria):
                result['Điểm tổng hợp'] += alt_weights[criterion] * crit_weights[i]
            result['Xếp hạng'] = result['Điểm tổng hợp'].rank(ascending=False, method='min').astype(int)
            result = result.sort_values('Xếp hạng').reset_index(drop=True)

            # --- Sau khi tổng hợp điểm các phương án (AHP), hiển thị bảng tổng hợp điểm các phương án ---
            st.markdown("### 🔢 Bảng tổng hợp điểm các phương án")
            st.dataframe(result)
            best_alt = result.iloc[0]['Phương án']
            best_score = result.iloc[0]['Điểm tổng hợp']
            st.success(f"Phương án có trọng số cao nhất là: **{best_alt}** (Điểm: {best_score:.4f})")
            # Vẽ biểu đồ cột so sánh điểm các phương án
            fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
            bars = ax_bar.bar(result['Phương án'], result['Điểm tổng hợp'], color=plt.get_cmap('Pastel1').colors)
            ax_bar.set_ylabel('Điểm tổng hợp')
            ax_bar.set_xlabel('Phương án')
            ax_bar.set_title('So sánh điểm các phương án', pad=20)
            ax_bar.set_ylim(0, result['Điểm tổng hợp'].max() + 0.1)  # tăng chiều cao trục Y

            for i, v in enumerate(result['Điểm tổng hợp']):
                ax_bar.text(i, v + 0.015, f"{v:.3f}", ha='center', va='bottom', fontsize=14, fontweight='bold')

            # Giảm padding trên để không che số
            plt.subplots_adjust(top=0.85)
            st.pyplot(fig_bar)

            # --- Tính các bảng trung gian cho tiêu chí ---
            ahp_matrix = st.session_state.ahp_matrix.values.astype(float)
            n_crit = len(selected_criteria)
            crit_col_sum = ahp_matrix.sum(axis=0)
            crit_sum_df = pd.DataFrame([crit_col_sum], columns=selected_criteria, index=["Sum"])
            crit_norm_matrix = ahp_matrix / crit_col_sum
            crit_norm_df = pd.DataFrame(crit_norm_matrix, columns=selected_criteria, index=selected_criteria)
            crit_weights_vec = crit_norm_matrix.mean(axis=1)
            crit_weights_df = pd.DataFrame({"Tiêu chí": selected_criteria, "Trọng số": crit_weights_vec})
            crit_weighted_sum = np.dot(ahp_matrix, crit_weights_vec)
            crit_consistency_vector = crit_weighted_sum / crit_weights_vec
            crit_ahp_detail_df = pd.DataFrame({
                'Tiêu chí': selected_criteria,
                'Weighted sum': crit_weighted_sum,
                'Trọng số': crit_weights_vec,
                'Consistency vector': crit_consistency_vector
            })
            crit_lambda_max = crit_consistency_vector.mean()
            crit_ci = (crit_lambda_max - n_crit) / (n_crit - 1)
            ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
            crit_ri = ri_table.get(n_crit, 1.49)
            crit_cr = crit_ci / crit_ri if crit_ri != 0 else 0
            crit_rank_df = crit_weights_df.copy()
            crit_rank_df['Rank'] = crit_rank_df['Trọng số'].rank(ascending=False, method='min').astype(int)
            crit_rank_df = crit_rank_df.sort_values('Rank')

            # --- Biểu đồ tròn cho trọng số tiêu chí và phương án ---
            fig = plt.figure(constrained_layout=True, figsize=(12, 6))
            gs = GridSpec(1, 2, figure=fig)
            # Pie chart for criteria
            ax1 = fig.add_subplot(gs[0, 0])
            wedges1, texts1, autotexts1 = ax1.pie(
                crit_weights_vec,
                labels=selected_criteria,
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.get_cmap('Set3').colors,
                textprops={'fontsize': 13, 'color': 'black'}
            )
            ax1.set_title('Trọng số các tiêu chí', fontsize=15)
            ax1.axis("equal")
            # Đặt legend xuống dưới biểu đồ
            ax1.legend(wedges1, selected_criteria, title="Tiêu chí", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=len(selected_criteria))
            # Pie chart for alternatives
            ax2 = fig.add_subplot(gs[0, 1])
            wedges2, texts2, autotexts2 = ax2.pie(
                result['Điểm tổng hợp'],
                labels=result['Phương án'],
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.get_cmap('Pastel1').colors,
                textprops={'fontsize': 13, 'color': 'black'}
            )
            ax2.set_title('Trọng số các phương án', fontsize=15)
            ax2.axis("equal")
            ax2.legend(wedges2, result['Phương án'], title="Phương án", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=len(result['Phương án']))
            plt.tight_layout()
            st.pyplot(fig)
            # Lưu hình ảnh biểu đồ
            chart_path = "ahp_pie_charts.png"
            fig.savefig(chart_path, bbox_inches='tight')

            # --- Xuất file Excel đầy đủ ---
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Tạo 1 sheet duy nhất cho tất cả ma trận
                all_matrix_rows = []
                # 1. Ma trận tiêu chí
                all_matrix_rows.append(['[Ma trận tiêu chí]'])
                crit_names = list(st.session_state.ahp_matrix.index)
                all_matrix_rows.append([''] + crit_names)
                for row in crit_names:
                    all_matrix_rows.append([row] + [st.session_state.ahp_matrix.loc[row, col] for col in crit_names])
                all_matrix_rows.append([])  # dòng trống
                # 2. Ma trận phương án cho từng tiêu chí
                for criterion in crit_names:
                    all_matrix_rows.append([f'[Ma trận phương án {criterion}]'])
                    alt_names = list(st.session_state[f"ahp_alt_matrix_{criterion}"].index)
                    all_matrix_rows.append([''] + alt_names)
                    for row in alt_names:
                        all_matrix_rows.append([row] + [st.session_state[f"ahp_alt_matrix_{criterion}"].loc[row, col] for col in alt_names])
                    all_matrix_rows.append([])
                # Ghi vào sheet duy nhất
                pd.DataFrame(all_matrix_rows).to_excel(writer, index=False, header=False, sheet_name='Ma trận tổng hợp')
                # (Các sheet khác giữ nguyên nếu muốn)
                # Tiêu chí và phương án đã chọn
                pd.DataFrame({'Tiêu chí': selected_criteria}).to_excel(writer, index=False, sheet_name='Tiêu chí đã chọn')
                pd.DataFrame({'Phương án': selected_alternatives}).to_excel(writer, index=False, sheet_name='Phương án đã chọn')
                # Tổng từng cột (Sum)
                crit_sum_df.to_excel(writer, sheet_name='Sum tiêu chí')
                # Ma trận chuẩn hóa
                crit_norm_df.to_excel(writer, sheet_name='Chuẩn hóa tiêu chí')
                # Trọng số tiêu chí
                crit_weights_df.to_excel(writer, index=False, sheet_name='Trọng số tiêu chí')
                # Bảng kiểm tra nhất quán
                crit_ahp_detail_df.to_excel(writer, index=False, sheet_name='Kiểm tra nhất quán tiêu chí')
                # Chỉ số nhất quán
                pd.DataFrame({
                    'Lambda_max': [crit_lambda_max],
                    'CI': [crit_ci],
                    'CR': [crit_cr]
                }).to_excel(writer, index=False, sheet_name='Chỉ số nhất quán tiêu chí')
                # Bảng xếp hạng tiêu chí
                crit_rank_df.to_excel(writer, index=False, sheet_name='Xếp hạng tiêu chí')
                # Kết quả tổng hợp
                result.to_excel(writer, index=False, sheet_name='Kết quả tổng hợp')
            st.download_button(
                label="📅 Tải kết quả Excel (.xlsx)",
                data=output.getvalue(),
                file_name="ahp_full_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # --- Xuất file PDF đầy đủ ---
            pdf = FPDF()
            pdf.add_font("DejaVu", style="", fname="dejavu-fonts-ttf-2.37/dejaVu-fonts-ttf-2.37/ttf/DejaVuSans.ttf", uni=True)
            pdf.add_font("DejaVu", style="B", fname="dejavu-fonts-ttf-2.37/dejaVu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf", uni=True)
            pdf.add_page()
            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="Báo cáo tổng hợp AHP", ln=1, align='C')
            pdf.ln()
            pdf.cell(200, 10, txt=f"Tiêu chí đã chọn: {', '.join(selected_criteria)}", ln=1)
            pdf.cell(200, 10, txt=f"Phương án đã chọn: {', '.join(selected_alternatives)}", ln=1)
            pdf.ln()
            # --- Ma trận tiêu chí ---
            pdf.set_font("DejaVu", style="B", size=12)
            pdf.cell(200, 10, txt="Ma trận tiêu chí", ln=1)
            pdf.set_font("DejaVu", size=11)
            # Tính chiều rộng cột động cho vừa trang
            ncol_crit = len(selected_criteria) + 1
            cell_w_crit = min(200 // ncol_crit, 40)
            pdf.cell(cell_w_crit, 8, "", border=1, align='C')
            for crit in selected_criteria:
                pdf.cell(cell_w_crit, 8, crit, border=1, align='C')
            pdf.ln()
            # Rows (lấy đúng số từ ma trận)
            ahp_matrix_arr = np.array(st.session_state.ahp_matrix)
            for i in range(len(selected_criteria)):
                pdf.cell(cell_w_crit, 8, selected_criteria[i], border=1, align='C')
                for j in range(len(selected_criteria)):
                    val = ahp_matrix_arr[i, j]
                    if isinstance(val, (int, float, np.floating)):
                        pdf.cell(cell_w_crit, 8, f"{val:.4f}", border=1, align='C')
                    else:
                        pdf.cell(cell_w_crit, 8, str(val), border=1, align='C')
                pdf.ln()
            # Tổng từng cột (Sum)
            pdf.set_font("DejaVu", style="B", size=11)
            pdf.cell(cell_w_crit, 8, "Sum", border=1, align='C')
            for v in crit_col_sum:
                pdf.cell(cell_w_crit, 8, f"{v:.4f}", border=1, align='C')
            pdf.ln()
            pdf.set_font("DejaVu", size=11)
            pdf.cell(200, 8, txt=f"Lambda_max: {crit_lambda_max:.4f}, CI: {crit_ci:.4f}, CR: {crit_cr:.4f}", ln=1)
            pdf.ln(2)
            # --- Trọng số các tiêu chí ---
            pdf.set_font("DejaVu", style="B", size=12)
            pdf.cell(200, 10, txt="Trọng số các tiêu chí", ln=1)
            pdf.set_font("DejaVu", size=11)
            pdf.cell(cell_w_crit, 8, "Tiêu chí", border=1, align='C')
            pdf.cell(cell_w_crit, 8, "Trọng số", border=1, align='C')
            pdf.ln()
            for i, row in crit_weights_df.iterrows():
                pdf.cell(cell_w_crit, 8, row['Tiêu chí'], border=1, align='C')
                pdf.cell(cell_w_crit, 8, f"{row['Trọng số']:.4f}", border=1, align='C')
                pdf.ln()
            # --- Bảng xếp hạng tiêu chí ---
            pdf.set_font("DejaVu", style="B", size=12)
            pdf.cell(200, 10, txt="Bảng xếp hạng tiêu chí", ln=1)
            pdf.set_font("DejaVu", size=11)
            pdf.cell(cell_w_crit, 8, "Tiêu chí", border=1, align='C')
            pdf.cell(cell_w_crit, 8, "Trọng số", border=1, align='C')
            pdf.cell(cell_w_crit, 8, "Rank", border=1, align='C')
            pdf.ln()
            for i, row in crit_rank_df.iterrows():
                pdf.cell(cell_w_crit, 8, row['Tiêu chí'], border=1, align='C')
                pdf.cell(cell_w_crit, 8, f"{row['Trọng số']:.4f}", border=1, align='C')
                pdf.cell(cell_w_crit, 8, f"{row['Rank']}", border=1, align='C')
                pdf.ln()
            pdf.ln(2)
            # --- Ma trận phương án cho từng tiêu chí ---
            for criterion in selected_criteria:
                pdf.set_font("DejaVu", style="B", size=12)
                pdf.cell(200, 10, txt=f"Ma trận phương án cho tiêu chí {criterion}", ln=1)
                pdf.set_font("DejaVu", size=11)
                ncol_alt = len(selected_alternatives) + 1
                cell_w_alt = min(200 // ncol_alt, 40)
                pdf.cell(cell_w_alt, 8, "", border=1, align='C')
                for alt in selected_alternatives:
                    pdf.cell(cell_w_alt, 8, alt, border=1, align='C')
                pdf.ln()
                alt_matrix = st.session_state[f"ahp_alt_matrix_{criterion}"].values.astype(float)
                for i in range(len(selected_alternatives)):
                    pdf.cell(cell_w_alt, 8, selected_alternatives[i], border=1, align='C')
                    for j in range(len(selected_alternatives)):
                        val = alt_matrix[i, j]
                        if isinstance(val, (int, float, np.floating)):
                            pdf.cell(cell_w_alt, 8, f"{val:.4f}", border=1, align='C')
                        else:
                            pdf.cell(cell_w_alt, 8, str(val), border=1, align='C')
                    pdf.ln()
                # Tổng từng cột (Sum)
                alt_col_sum = alt_matrix.sum(axis=0)
                pdf.set_font("DejaVu", style="B", size=11)
                pdf.cell(cell_w_alt, 8, "Sum", border=1, align='C')
                for v in alt_col_sum:
                    pdf.cell(cell_w_alt, 8, f"{v:.4f}", border=1, align='C')
                pdf.ln()
                pdf.set_font("DejaVu", size=11)
                # Chỉ số nhất quán
                alt_norm_matrix = alt_matrix / alt_col_sum
                alt_weights_vec = alt_norm_matrix.mean(axis=1)
                alt_weighted_sum = np.dot(alt_matrix, alt_weights_vec)
                alt_consistency_vector = alt_weighted_sum / alt_weights_vec
                alt_lambda_max = alt_consistency_vector.mean()
                alt_ci = (alt_lambda_max - len(selected_alternatives)) / (len(selected_alternatives) - 1)
                alt_ri = ri_table.get(len(selected_alternatives), 1.49)
                alt_cr = alt_ci / alt_ri if alt_ri != 0 else 0
                pdf.cell(200, 8, txt=f"Lambda_max: {alt_lambda_max:.4f}, CI: {alt_ci:.4f}, CR: {alt_cr:.4f}", ln=1)
                pdf.ln(2)
            # --- Kết quả tổng hợp ---
            pdf.set_font("DejaVu", style="B", size=12)
            pdf.cell(200, 10, txt="Kết quả tổng hợp", ln=1)
            pdf.set_font("DejaVu", size=11)
            ncol_res = len(selected_criteria) + 3
            cell_w_res = min(200 // ncol_res, 40)
            pdf.cell(cell_w_res, 8, "Phương án", border=1, align='C')
            for crit in selected_criteria:
                pdf.cell(cell_w_res, 8, crit, border=1, align='C')
            pdf.cell(cell_w_res, 8, "Điểm tổng hợp", border=1, align='C')
            pdf.cell(cell_w_res, 8, "Xếp hạng", border=1, align='C')
            pdf.ln()
            for i, row in result.iterrows():
                pdf.cell(cell_w_res, 8, row['Phương án'], border=1, align='C')
                for crit in selected_criteria:
                    val = row[crit]
                    if isinstance(val, (int, float, np.floating)):
                        pdf.cell(cell_w_res, 8, f"{val:.4f}", border=1, align='C')
                    else:
                        pdf.cell(cell_w_res, 8, str(val), border=1, align='C')
                pdf.cell(cell_w_res, 8, f"{row['Điểm tổng hợp']:.4f}", border=1, align='C')
                pdf.cell(cell_w_res, 8, f"{row['Xếp hạng']}", border=1, align='C')
                pdf.ln()
            # Thêm hình ảnh biểu đồ
            if os.path.exists(chart_path):
                pdf.image(chart_path, x=10, w=190)
            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'replace')
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)
            st.download_button(
                label="📄 Tải tổng hợp PDF",
                data=pdf_output,
                file_name="ahp_full_result.pdf",
                mime="application/pdf"
            )

            # Thêm nút lưu phiên làm việc
            if st.button("💾 Lưu phiên làm việc"):
                try:
                    # Lưu kết quả vào database
                    save_evaluation_session(
                        username=username,
                        criteria=selected_criteria,
                        alternatives=selected_alternatives,
                        ahp_matrix=st.session_state.ahp_matrix.values.astype(float),
                        topsis_scores=result.set_index('Phương án').to_dict('index'),
                        final_results=result.set_index('Phương án').to_dict('index')
                    )
                    st.success("✅ Đã lưu phiên làm việc thành công!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi lưu phiên làm việc: {e}")

        

