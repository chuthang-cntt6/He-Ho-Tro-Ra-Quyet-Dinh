# Hướng Dẫn Cài Đặt Dự Án AHP-TOPSIS

## Bước 1: Tải Dự Án
1. Truy cập vào đường dẫn GitHub của dự án: [Link GitHub]
2. Nhấn vào nút "Code" màu xanh lá
3. Chọn "Download ZIP" để tải toàn bộ dự án về máy
4. Giải nén file ZIP vừa tải về vào thư mục mong muốn

## Bước 2: Cài Đặt Python
1. Truy cập trang web [python.org](https://www.python.org/downloads/)
2. Tải phiên bản Python mới nhất (khuyến nghị Python 3.8 trở lên)
3. Trong quá trình cài đặt, đảm bảo tích chọn "Add Python to PATH"
4. Kiểm tra cài đặt bằng cách mở Command Prompt và gõ:
   ```
   python --version
   ```

## Bước 3: Cài Đặt Các Thư Viện Cần Thiết
1. Mở Command Prompt
2. Di chuyển đến thư mục dự án:
   ```
   cd đường_dẫn_đến_thư_mục_dự_án
   ```
3. Cài đặt các thư viện cần thiết:
   ```
   pip install streamlit
   pip install pandas
   pip install numpy
   pip install matplotlib
   pip install seaborn
   pip install openpyxl
   ```

## Bước 4: Chạy Ứng Dụng
1. Trong Command Prompt, đảm bảo bạn đang ở thư mục dự án
2. Chạy lệnh:
   ```
   streamlit run streamlit_app.py
   ```
3. Ứng dụng sẽ tự động mở trong trình duyệt web mặc định của bạn

## Cấu Trúc Dự Án
- `streamlit_app.py`: File chính chứa giao diện người dùng
- `database.py`: Quản lý cơ sở dữ liệu SQLite
- `ahp.py`: Triển khai phương pháp AHP
- `topsis.py`: Triển khai phương pháp TOPSIS
- `ahp_topsis.db`: Cơ sở dữ liệu SQLite
- `AHP_Van_Chuyen.xlsx`: File dữ liệu mẫu

## Lưu Ý
- Đảm bảo máy tính có kết nối internet để tải các thư viện cần thiết
- Nếu gặp lỗi khi cài đặt thư viện, thử chạy lệnh với quyền Administrator
- Đảm bảo đã cài đặt đầy đủ các thư viện trước khi chạy ứng dụng 