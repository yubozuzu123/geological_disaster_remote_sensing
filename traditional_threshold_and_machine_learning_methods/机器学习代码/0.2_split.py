import pandas as pd
from sklearn.model_selection import train_test_split

# 文件路径
file_path = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\2015feature.csv"

# 读取数据
df = pd.read_csv(file_path, encoding='utf-8')

print(f"原始数据形状：{df.shape}")

# 假设最后一列是标签
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 拆分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 合并
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
# 保存为 Excel
train_df.to_excel(r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\train.xlsx", index=False)
test_df.to_excel(r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\test.xlsx", index=False)

print(f"拆分完成：训练集 {train_df.shape[0]} 行，测试集 {test_df.shape[0]} 行")
print("训练集保存为：train.xlsx")
print("测试集保存为：test.xlsx")
