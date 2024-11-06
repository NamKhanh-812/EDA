# load_data.py
import pandas as pd

def load_data(file_path):
    """
    Hàm tải dữ liệu từ file CSV và kiểm tra thông tin cơ bản.
    """
    df = pd.read_csv(file_path)
    print("Dữ liệu đã tải thành công!")
    print("Kích thước dữ liệu:", df.shape)
    print("Thông tin dữ liệu:")
    print(df.info())
    return df



from load_data import load_data

file_path = "F:/TTS/EDA/fifa_players.csv"  # Thay bằng đường dẫn thực tế của bạn
df = load_data(file_path)
