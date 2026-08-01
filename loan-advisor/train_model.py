import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# قراءة البيانات
df = pd.read_csv('data/lending_club_cleaned.csv')
print(f"📊 حجم البيانات: {df.shape}")

# 1. تحويل المتغيرات النصية إلى أرقام
categorical_cols = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 2. اختيار الميزات للتدريب
feature_cols = [
    'loan_amnt', 'int_rate', 'installment', 'dti',
    'delinq_2yrs', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'fico_score', 'emp_length_clean',
    'term_encoded', 'grade_encoded', 'home_ownership_encoded',
    'verification_status_encoded', 'purpose_encoded'
]

X = df[feature_cols]
y = df['loan_status_clean'].map({'Good': 0, 'Bad': 1})

print(f"✅ عدد الميزات: {X.shape[1]}")
print(f"🎯 توزيع الهدف: {y.value_counts().to_dict()}")

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 حجم التدريب: {len(X_train)}")
print(f"📊 حجم الاختبار: {len(X_test)}")

# 4. تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. تدريب النموذج
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 6. تقييم النموذج
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 تقييم النموذج:")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nتقرير التصنيف:")
print(classification_report(y_test, y_pred))

# مصفوفة الارتباك
cm = confusion_matrix(y_test, y_pred)
print("\n📋 مصفوفة الارتباك:")
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")

# 7. حفظ النموذج
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/features.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print("\n💾 تم حفظ النموذج في مجلد models/")
