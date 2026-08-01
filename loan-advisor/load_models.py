import joblib
import pandas as pd

print("📂 جاري تحميل الملفات المحفوظة...\n")

# 1. تحميل النموذج
model = joblib.load('models/loan_model.pkl')
print(f"✅ تم تحميل النموذج: {type(model).__name__}")

# 2. تحميل المقياس
scaler = joblib.load('models/scaler.pkl')
print(f"✅ تم تحميل المقياس: {type(scaler).__name__}")

# 3. تحميل الميزات
features = joblib.load('models/features.pkl')
print(f"✅ تم تحميل {len(features)} ميزة:")
for i, f in enumerate(features, 1):
    print(f"   {i}. {f}")

# 4. تحميل الترميزات
label_encoders = joblib.load('models/label_encoders.pkl')
print(f"\n✅ تم تحميل {len(label_encoders)} ترميز:")
for key in label_encoders.keys():
    print(f"   - {key}")

# 5. عرض أهمية الميزات
print("\n📊 أهمية الميزات (أعلى 5):")
importances = model.feature_importances_
sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
for i in sorted_idx[:5]:
    print(f"   {features[i]}: {importances[i]:.4f}")

# 6. اختبار النموذج على بيانات وهمية
print("\n🧪 اختبار النموذج على بيانات وهمية:")
sample_data = [[
    20000,   # loan_amnt
    10.5,    # int_rate
    650,     # installment
    15.0,    # dti
    0,       # delinq_2yrs
    5,       # open_acc
    0,       # pub_rec
    10000,   # revol_bal
    30.0,    # revol_util
    10,      # total_acc
    700,     # fico_score
    5,       # emp_length_clean
    0,       # term_encoded
    4,       # grade_encoded
    1,       # home_ownership_encoded
    0,       # verification_status_encoded
    2        # purpose_encoded
]]

# تطبيع البيانات
sample_scaled = scaler.transform(sample_data)

# التنبؤ
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print(f"   احتمالية التعثر: {probability*100:.2f}%")
print(f"   التصنيف: {'سيء' if prediction == 1 else 'جيد'}")
