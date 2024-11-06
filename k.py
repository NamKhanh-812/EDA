import pandas as pd

file_path = 'F:/TTS/EDA/fifa_players.csv'  # Đảm bảo đường dẫn đúng
df = pd.read_csv(file_path)

# In ra các tên cột trong DataFrame
print(df.columns)
