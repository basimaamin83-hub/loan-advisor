import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# تحديد مسار الملف (نفس المسار الذي حمّلته سابقاً)
# استخدم المسار الذي ظهر في تشغيل load_lending_club.py
file_path = os.path.expanduser("~/.cache/kagglehub/datasets/wordsforthewise/lending-club/versions/3/accepted_2007_to_2018Q4.csv.gz")

print("📊 جاري تحميل البيانات...")
df = pd.read_csv(file_path, nrows=10000)

print(f"📊 حجم البيانات قبل التنظيف: {df.shape}")

# 1. اختيار الأعمدة المهمة
important_columns = [
    'loan_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'verification_status', 'purpose',
    'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'loan_status'
]

# استخراج الأعمدة المهمة
df_clean = df[important_columns].copy()
print(f"✅ تم اختيار {len(df_clean.columns)} عمود من أصل {len(df.columns)}")

# 2. تنظيف عمود loan_status (الهدف)
def clean_loan_status(status):
    if pd.isna(status):
        return np.nan
    status_str = str(status).lower()
    good_statuses = ['fully paid', 'paid', 'current']
    if any(good in status_str for good in good_statuses):
        return 'Good'
    bad_statuses = ['charged off', 'default', 'late', 'in grace period']
    if any(bad in status_str for bad in bad_statuses):
        return 'Bad'
    return 'Good'

df_clean['loan_status_clean'] = df_clean['loan_status'].apply(clean_loan_status)

# 3. تنظيف نسبة الفائدة (إزالة %)
df_clean['int_rate'] = pd.to_numeric(
    df_clean['int_rate'].astype(str).str.replace('%', '', regex=False),
    errors='coerce'
)

# 4. تنظيف استخدام الرصيد (إزالة %)
df_clean['revol_util'] = pd.to_numeric(
    df_clean['revol_util'].astype(str).str.replace('%', '', regex=False),
    errors='coerce'
)

# 5. تنظيف مدة العمل
def clean_emp_length(value):
    if pd.isna(value):
        return 0
    value = str(value).lower().strip()
    if '10+' in value:
        return 10
    if '< 1' in value:
        return 0
    import re
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

df_clean['emp_length_clean'] = df_clean['emp_length'].apply(clean_emp_length)

# 6. حساب متوسط درجة FICO
df_clean['fico_score'] = (df_clean['fico_range_low'] + df_clean['fico_range_high']) / 2

# 7. إزالة القيم المفقودة
print(f"🔍 القيم المفقودة قبل الحذف: {df_clean.isnull().sum().sum()}")
df_clean = df_clean.dropna()
print(f"✅ بعد الحذف: {len(df_clean)} صف")

# 8. إنشاء مجلد data إذا لم يكن موجوداً
os.makedirs('data', exist_ok=True)

# 9. حفظ البيانات النظيفة
df_clean.to_csv('data/lending_club_cleaned.csv', index=False)
print("💾 تم حفظ البيانات النظيفة في data/lending_club_cleaned.csv")

print("\n📊 توزيع الهدف:")
print(df_clean['loan_status_clean'].value_counts())