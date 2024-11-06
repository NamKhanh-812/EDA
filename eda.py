# eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(df):
    """
    Hàm thực hiện phân tích khám phá dữ liệu (EDA) với các biểu đồ.
    """
    # Kiểm tra và xử lý cột 'Overall rating'
    if 'Overall rating' in df.columns:
        df['Overall rating'] = pd.to_numeric(df['Overall rating'], errors='coerce')
        
        # Biểu đồ phân phối của 'Overall rating'
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Overall rating'], kde=True)
        plt.title("Phân phối của Overall Rating")
        plt.xlabel("Overall Rating")
        plt.show()
    else:
        print("Cột 'Overall rating' không tồn tại trong dữ liệu.")
    
    # Lọc các cột số và tính ma trận tương quan
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Ma trận tương quan")
    plt.show()
