# data_cleaning.py
def clean_data(df):
    """
    Hàm xử lý dữ liệu bị thiếu bằng cách loại bỏ các hàng chứa giá trị bị thiếu.
    """
    print("Số lượng giá trị thiếu trước khi làm sạch:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Loại bỏ các hàng chứa giá trị thiếu
    df = df.dropna()
    print("Số lượng giá trị thiếu sau khi làm sạch:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df



from data_cleaning import clean_data

df = clean_data(df)
