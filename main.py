# # main.py
# import pandas as pd
# from eda import perform_eda

# # Tải dữ liệu
# file_path = "F:/TTS/EDA/fifa_players.csv"  # Thay bằng đường dẫn thực tế của bạn
# df = pd.read_csv(file_path)

# # Thực hiện EDA
# perform_eda(df)



#      2

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Đường dẫn tới tệp CSV
# file_path = "F:/TTS/EDA/fifa_players.csv"

# # Tải dữ liệu
# df = pd.read_csv(file_path)

# # Tiền xử lý dữ liệu
# df.columns = df.columns.str.strip()  # Loại bỏ khoảng trắng thừa trong tên cột

# # Các cột tính toán và mục tiêu
# features = ['Age', 'Potential']
# target = 'Overall rating'

# # Kiểm tra số lượng mẫu ban đầu
# print(f"Số mẫu ban đầu: {len(df)}")

# # Loại bỏ các hàng có giá trị trống trong các cột quan trọng
# df = df.dropna(subset=features + [target])

# # Kiểm tra số lượng mẫu sau khi loại bỏ NaN
# print(f"Số mẫu sau khi loại bỏ NaN: {len(df)}")

# # Kiểm tra các giá trị không hợp lệ trong các cột và chuyển đổi nếu cần
# df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# df['Potential'] = pd.to_numeric(df['Potential'], errors='coerce')
# df['Overall rating'] = pd.to_numeric(df['Overall rating'], errors='coerce')

# # Loại bỏ các hàng có giá trị NaN sau khi chuyển đổi
# df = df.dropna(subset=features + [target])

# # Tính toán mode và median của Overall rating
# mode_value = df[target].mode()[0]
# median_value = df[target].median()

# print(f"Giá trị thường gặp nhất (mode) của Overall rating: {mode_value}")
# print(f"Trung vị (median) của Overall rating: {median_value}")

# # Vẽ biểu đồ phân phối của Overall rating
# plt.figure(figsize=(10,6))
# sns.histplot(df[target], bins=50, kde=True)
# plt.title('Phân phối của Overall rating')
# plt.xlabel('Overall rating')
# plt.ylabel('Tần suất')

# # Vẽ các đường dọc cho mode và median
# plt.axvline(mode_value, color='red', linestyle='--', label=f'Mode: {mode_value}')
# plt.axvline(median_value, color='blue', linestyle='--', label=f'Median: {median_value}')

# # Dự đoán Overall rating cho cầu thủ nhập vào
# player_data = {
#     'Age': 25, 
#     'Potential': 85  # Bỏ cột 'Value' ở đây
# }

# # Huấn luyện mô hình RandomForestRegressor
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Dự đoán Overall rating cho cầu thủ
# predicted_rating = model.predict(pd.DataFrame([player_data]))[0]
# print(f"Dự đoán Overall rating: {predicted_rating}")

# # Vẽ đường dọc cho predicted rating
# plt.axvline(predicted_rating, color='green', linestyle='--', label=f'Predicted: {predicted_rating:.2f}')

# # Thêm chú thích cho biểu đồ
# plt.legend()

# # Hiển thị biểu đồ
# plt.show()



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đường dẫn tới tệp CSV
file_path = "F:/TTS/EDA/fifa_players.csv"

# Tải dữ liệu
df = pd.read_csv(file_path)

# Tiền xử lý dữ liệu
df.columns = df.columns.str.strip()  # Loại bỏ khoảng trắng thừa trong tên cột

# Chọn các cột đặc trưng (features) và mục tiêu (target)
features = ['Age', 'Potential', 'Height', 'Weight', 'Growth', 'Wage', 'Release clause', 
            'Total attacking', 'Crossing', 'Finishing', 'Heading accuracy', 'Short passing', 'Volleys',
            'Total skill', 'Dribbling', 'Curve', 'FK Accuracy', 'Long passing', 'Ball control', 
            'Total movement', 'Acceleration', 'Sprint speed', 'Agility', 'Reactions', 'Balance', 
            'Total power', 'Shot power', 'Jumping', 'Stamina', 'Strength', 'Long shots', 'Total mentality',
            'Aggression', 'Interceptions', 'Att. Position', 'Vision', 'Penalties', 'Composure', 
            'Total defending', 'Defensive awareness', 'Standing tackle', 'Sliding tackle', 
            'Total goalkeeping', 'GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning', 'GK Reflexes',
            'Total stats', 'Base stats', 'Weak foot', 'Skill moves', 'Attacking work rate', 'Defensive work rate',
            'International reputation', 'Body type', 'Real face', 'Pace / Diving', 'Shooting / Handling', 
            'Passing / Kicking', 'Dribbling / Reflexes', 'Defending / Pace', 'Physical / Positioning']
target = 'Overall rating'

# Kiểm tra số lượng mẫu ban đầu
print(f"Số mẫu ban đầu: {len(df)}")

# Loại bỏ các hàng có giá trị trống trong các cột quan trọng
df = df.dropna(subset=features + [target])

# Kiểm tra số lượng mẫu sau khi loại bỏ NaN
print(f"Số mẫu sau khi loại bỏ NaN: {len(df)}")

# Chuyển đổi các cột tiền tệ và các cột số
df['Wage'] = df['Wage'].apply(lambda x: str(x).replace('€', '').replace('K', 'e3').replace('M', 'e6'))  # Chuyển đổi wage
df['Wage'] = pd.to_numeric(df['Wage'], errors='coerce')  # Chuyển thành giá trị số

df['Release clause'] = df['Release clause'].apply(lambda x: str(x).replace('€', '').replace('K', 'e3').replace('M', 'e6'))  # Chuyển đổi release clause
df['Release clause'] = pd.to_numeric(df['Release clause'], errors='coerce')  # Chuyển thành giá trị số

df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Potential'] = pd.to_numeric(df['Potential'], errors='coerce')
df['Strength'] = pd.to_numeric(df['Strength'], errors='coerce')
df['Dribbling'] = pd.to_numeric(df['Dribbling'], errors='coerce')
df['Sprint speed'] = pd.to_numeric(df['Sprint speed'], errors='coerce')
df['Overall rating'] = pd.to_numeric(df['Overall rating'], errors='coerce')

# Loại bỏ các hàng có giá trị NaN sau khi chuyển đổi
df = df.dropna(subset=features + [target])

# Huấn luyện mô hình RandomForestRegressor
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán Overall rating cho tất cả các cầu thủ trong tập kiểm tra
predicted_ratings = model.predict(X_test)

# In một số kết quả dự đoán và thực tế
print(f"Predicted Overall Ratings for some players: {predicted_ratings[:5]}")

# In giá trị thực tế của Overall rating
print(f"True Overall Ratings for some players: {y_test[:5].values}")

# Tính toán và hiển thị các chỉ số thống kê mô tả cho Overall rating
plt.figure(figsize=(10,6))

# Vẽ biểu đồ phân phối của các giá trị dự đoán và giá trị thực tế
plt.hist(y_test, bins=50, alpha=0.5, label='True Overall Rating')
plt.hist(predicted_ratings, bins=50, alpha=0.5, label='Predicted Overall Rating')

# Vẽ giá trị mode và median
mode = y_test.mode()[0]
median = np.median(y_test)

plt.axvline(mode, color='r', linestyle='dashed', linewidth=2, label=f'Mode: {mode}')
plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median}')

# Thêm tiêu đề và nhãn
plt.title('Distribution of True and Predicted Overall Rating')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.legend()

# Hiển thị biểu đồ
plt.show()
