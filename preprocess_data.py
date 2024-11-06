# preprocess_data.py
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Hàm chuẩn hóa và phân chia dữ liệu thành tập huấn luyện và kiểm tra.
    """
    features = df[['age', 'potential', 'dribbling', 'passing', 'shooting']]  # Các đặc trưng chính
    target = df['Overall rating']
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    print("Kích thước tập huấn luyện:", X_train.shape)
    print("Kích thước tập kiểm tra:", X_test.shape)
    return X_train, X_test, y_train, y_test


from preprocess_data import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data(df)
