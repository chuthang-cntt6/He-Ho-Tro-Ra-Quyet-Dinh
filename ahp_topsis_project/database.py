import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    """Khởi tạo cơ sở dữ liệu và các bảng cần thiết"""
    conn = sqlite3.connect('ahp_topsis.db')
    c = conn.cursor()
    
    # Bảng tiêu chí
    c.execute('''
        CREATE TABLE IF NOT EXISTS criteria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Bảng phương án
    c.execute('''
        CREATE TABLE IF NOT EXISTS alternatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Bảng phiên đánh giá
    c.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    ''')
    
    # Bảng ma trận AHP
    c.execute('''
        CREATE TABLE IF NOT EXISTS ahp_matrices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            criterion1_id INTEGER,
            criterion2_id INTEGER,
            value REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES evaluation_sessions (id),
            FOREIGN KEY (criterion1_id) REFERENCES criteria (id),
            FOREIGN KEY (criterion2_id) REFERENCES criteria (id)
        )
    ''')
    
    # Bảng điểm TOPSIS
    c.execute('''
        CREATE TABLE IF NOT EXISTS topsis_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            alternative_id INTEGER,
            criterion_id INTEGER,
            score REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES evaluation_sessions (id),
            FOREIGN KEY (alternative_id) REFERENCES alternatives (id),
            FOREIGN KEY (criterion_id) REFERENCES criteria (id)
        )
    ''')
    
    # Bảng kết quả cuối cùng
    c.execute('''
        CREATE TABLE IF NOT EXISTS final_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            alternative_id INTEGER,
            topsis_score REAL NOT NULL,
            rank INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES evaluation_sessions (id),
            FOREIGN KEY (alternative_id) REFERENCES alternatives (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_default_data():
    """Thêm dữ liệu mặc định cho tiêu chí và phương án"""
    conn = sqlite3.connect('ahp_topsis.db')
    c = conn.cursor()
    
    # Thêm tiêu chí mặc định
    default_criteria = [
        ('Chi phí', 'Chi phí vận chuyển'),
        ('Thời gian', 'Thời gian vận chuyển'),
        ('Ổn định', 'Mức độ ổn định của phương án'),
        ('An toàn', 'Mức độ an toàn'),
        ('Linh hoạt', 'Khả năng linh hoạt trong vận chuyển')
    ]
    
    c.executemany('''
        INSERT OR IGNORE INTO criteria (name, description)
        VALUES (?, ?)
    ''', default_criteria)
    
    # Thêm phương án mặc định
    default_alternatives = [
        ('Đường bộ', 'Vận chuyển bằng xe tải, container'),
        ('Đường biển', 'Vận chuyển bằng tàu biển'),
        ('Hàng không', 'Vận chuyển bằng máy bay'),
        ('Đường sắt', 'Vận chuyển bằng tàu hỏa')
    ]
    
    c.executemany('''
        INSERT OR IGNORE INTO alternatives (name, description)
        VALUES (?, ?)
    ''', default_alternatives)
    
    conn.commit()
    conn.close()

def get_all_criteria():
    """Lấy danh sách tất cả tiêu chí"""
    conn = sqlite3.connect('ahp_topsis.db')
    df = pd.read_sql_query("SELECT * FROM criteria", conn)
    conn.close()
    return df

def get_all_alternatives():
    """Lấy danh sách tất cả phương án"""
    conn = sqlite3.connect('ahp_topsis.db')
    df = pd.read_sql_query("SELECT * FROM alternatives", conn)
    conn.close()
    return df

def add_criterion(name, description):
    """Thêm một tiêu chí mới vào cơ sở dữ liệu"""
    conn = sqlite3.connect('ahp_topsis.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO criteria (name, description)
            VALUES (?, ?)
        ''', (name, description))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def add_alternative(name, description):
    """Thêm một phương án mới vào cơ sở dữ liệu"""
    conn = sqlite3.connect('ahp_topsis.db')
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO alternatives (name, description)
            VALUES (?, ?)
        ''', (name, description))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def save_evaluation_session(username, criteria, alternatives, ahp_matrix, topsis_scores, final_results):
    """Lưu một phiên đánh giá mới"""
    conn = sqlite3.connect('ahp_topsis.db')
    c = conn.cursor()
    
    # Tạo phiên đánh giá mới
    c.execute('''
        INSERT INTO evaluation_sessions (username)
        VALUES (?)
    ''', (username,))
    session_id = c.lastrowid
    
    # Lưu ma trận AHP
    for i, crit1 in enumerate(criteria):
        for j, crit2 in enumerate(criteria):
            c.execute('''
                INSERT INTO ahp_matrices (session_id, criterion1_id, criterion2_id, value)
                VALUES (?, ?, ?, ?)
            ''', (session_id, crit1, crit2, ahp_matrix[i][j]))
    
    # Lưu điểm TOPSIS
    for alt_id, scores in topsis_scores.items():
        for crit_id, score in scores.items():
            c.execute('''
                INSERT INTO topsis_scores (session_id, alternative_id, criterion_id, score)
                VALUES (?, ?, ?, ?)
            ''', (session_id, alt_id, crit_id, score))
    
    # Lưu kết quả cuối cùng
    for alt_id, result in final_results.items():
        c.execute('''
            INSERT INTO final_results (session_id, alternative_id, topsis_score, rank)
            VALUES (?, ?, ?, ?)
        ''', (
            session_id,
            alt_id,
            result.get('Điểm TOPSIS', result.get('score', 0)),
            result.get('Xếp hạng', result.get('rank', 0))
        ))
    
    conn.commit()
    conn.close()

def get_evaluation_history(username):
    """Lấy lịch sử các phiên đánh giá của người dùng"""
    conn = sqlite3.connect('ahp_topsis.db')
    df = pd.read_sql_query('''
        SELECT es.*, 
               COUNT(DISTINCT fr.alternative_id) as num_alternatives,
               COUNT(DISTINCT am.criterion1_id) as num_criteria
        FROM evaluation_sessions es
        LEFT JOIN final_results fr ON es.id = fr.session_id
        LEFT JOIN ahp_matrices am ON es.id = am.session_id
        WHERE es.username = ?
        GROUP BY es.id
        ORDER BY es.created_at DESC
    ''', conn, params=(username,))
    conn.close()
    return df

# Khởi tạo cơ sở dữ liệu khi import module
init_db()
add_default_data() 