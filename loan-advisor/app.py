import pandas as pd
from sklearn.datasets import fetch_openml

print("📥 جاري تحميل البيانات...")
df = fetch_openml(data_id=31, as_frame=True).frame

df.to_csv('data/german_credit.csv', index=False)
print(f"✅ تم حفظ البيانات في data/german_credit.csv")
print(f"📊 عدد الصفوف: {len(df)}")
print(f"📋 عدد الأعمدة: {len(df.columns)}")

print("\n🔍 أول 5 صفوف:")
print(df.head())