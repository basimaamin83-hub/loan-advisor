import pandas as pd
import kagglehub
import os

os.environ['KAGGLE_API_TOKEN'] = "KGAT_d6a624559220540587057a55849fa582"

print("📥 Loading Lending Club dataset...")
path = kagglehub.dataset_download("wordsforthewise/lending-club")
file_path = os.path.join(path, "accepted_2007_to_2018Q4.csv.gz")
print(f"📍 File path: {file_path}")

print("📖 Reading sample data...")
df_sample = pd.read_csv(file_path, nrows=10000)

print(f"✅ Loaded {len(df_sample)} rows successfully!")
print("\n🔍 First 5 rows:")
print(df_sample.head())
print("\n📊 Column names:")
print(df_sample.columns.tolist())

# معلومات إضافية عن البيانات
print("\n📊 معلومات عامة:")
print(f"عدد الصفوف: {len(df_sample)}")
print(f"عدد الأعمدة: {len(df_sample.columns)}")
print(f"الذاكرة المستخدمة: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} ميجابايت")

print("\n📈 إحصائيات الأعمدة الرقمية:")
print(df_sample.describe())

print("\n🔍 أنواع الأعمدة:")
print(df_sample.dtypes.value_counts())
