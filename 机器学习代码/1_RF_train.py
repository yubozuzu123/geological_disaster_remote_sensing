
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib  # 保存和加载模型
import json  # 保存参数

# 数据文件路径
train_file = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\train.xlsx"
test_file = r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\test.xlsx"

# 保存文件路径
param_save_path =  r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\训练结果\best_rf_params2.json"
model_save_path =  r"F:\滑坡\2026.3.25ppt制作\上课\培训材料\影像及标签\训练结果\best_rf_model2.pkl"

# 加载数据函数
def load_data(file_path):
    """
    从 Excel 或 CSV 文件加载数据，将字符串标签转换为数值
    """
    data = pd.read_excel(file_path)  # 或 pd.read_csv，根据实际格式
    features = data.iloc[:, :-1]      # 前 n-1 列为特征
    labels = data.iloc[:, -1]         # 最后一列为标签

    # 将字符串标签映射为整数
    label_mapping = {'landslide': 1, 'non-landslide': 0}
    labels = labels.map(label_mapping)

    # 处理缺失值（根据情况选择填充值，例如 0）
    features = features.fillna(0)
    labels = labels.fillna(0).astype(int)

    return features, labels

# 加载数据
X_train, y_train = load_data(train_file)
X_test, y_test = load_data(test_file)

# 确保没有列名：将 DataFrame 转为 NumPy 数组
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# 网格搜索参数
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# 网格搜索
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=1)
grid_search.fit(X_train, y_train)

# 保存最佳参数
best_params = grid_search.best_params_
with open(param_save_path, 'w') as f:
    json.dump(best_params, f)

# 使用最佳参数重新训练
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# 保存模型
joblib.dump(best_rf, model_save_path)

# 预测
y_pred = best_rf.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

# 打印结果
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

