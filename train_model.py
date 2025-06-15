import os
import shutil
import kagglehub
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# تحميل البيانات من KaggleHub
path = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
print("Path to dataset files:", path)

# تحديد المسار الجديد
target_path = "./dtcar"
os.makedirs(target_path, exist_ok=True)
for file_name in os.listdir(path):
    full_file_path = os.path.join(path, file_name)
    shutil.copy(full_file_path, target_path)

print("✔️ تم نسخ الملفات إلى:", target_path)

# تحميل البيانات
df = pd.read_csv("./dtcar/used_cars.csv")

# تنظيف البيانات
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
df = df[(df['price'] >= 3000) & (df['price'] <= 100000)]
df = df[(df['milage'] < 250000)]

df['car_age'] = 2025 - df['model_year']

def extract_engine_size(engine_str):
    match = re.search(r'(\d+\.\d+)L', str(engine_str))
    return float(match.group(1)) if match else np.nan

df['engine_size'] = df['engine'].apply(extract_engine_size)
df['engine_size'].fillna(df['engine_size'].median(), inplace=True)

df['milage_log'] = np.log1p(df['milage'])
df['mileage_per_year'] = df['milage'] / (df['car_age'] + 1)
df['price_per_mile'] = df['price'] / (df['milage'] + 1)
df['age_mileage_interaction'] = df['car_age'] * df['mileage_per_year']

top_models = df['model'].value_counts().nlargest(50).index
df['model'] = df['model'].where(df['model'].isin(top_models), other='Other')
df['model_freq'] = df['model'].map(df['model'].value_counts())

categorical_features = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
numerical_features = [
    'car_age', 'milage_log', 'engine_size',
    'mileage_per_year', 'price_per_mile',
    'age_mileage_interaction', 'model_freq'
]

X = df[categorical_features + numerical_features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# تدريب XGBoost
xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    early_stopping_rounds=30,
    eval_metric='mae',
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)], verbose=False)

# تدريب LightGBM
lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm.fit(X_train_processed, y_train)

# حفظ النماذج والمعالج
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(lgbm, 'lgbm_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ تم حفظ النماذج والمعالج بنجاح.")
