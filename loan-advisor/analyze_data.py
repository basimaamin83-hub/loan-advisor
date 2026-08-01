import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# قراءة البيانات النظيفة
df = pd.read_csv('data/lending_club_cleaned.csv')

print("📊 معلومات عامة عن البيانات النظيفة:")
print(f"عدد الصفوف: {len(df)}")
print(f"عدد الأعمدة: {len(df.columns)}")
print("\n🔍 أول 5 صفوف:")
print(df.head())

print("\n📈 إحصائيات الأعمدة الرقمية:")
print(df.describe())

print("\n📊 توزيع الهدف:")
print(df['loan_status_clean'].value_counts())

# تحليل العلاقة بين المتغيرات والهدف
print("\n📊 متوسط القيم حسب حالة القرض:")
print(df.groupby('loan_status_clean')[['loan_amnt', 'int_rate', 'dti', 'fico_score']].mean())
