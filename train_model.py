# train_model.py
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """
    Hàm huấn luyện mô hình hồi quy tuyến tính.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Mô hình đã huấn luyện thành công!")
    return model


from train_model import train_model

model = train_model(X_train, y_train)
