import joblib
import pandas as pd

model=joblib.load('models/loan_model.pkl')
scaler=joblib.load('models/scaler.pkl')
features=joblib.load('models/features.pkl')
print('Loaded', type(model), type(scaler), 'features_len', len(features))
loan_amnt=20000
int_rate=12.0
monthly_payment=664
dti=15
delinq_2yrs=0
open_acc=5
pub_rec=0
revol_bal=10000
revol_util=30
total_acc=10
fico_score=700
emp_length=5
term_encoded=0
grade_encoded=0
home_ownership_encoded=0
purpose_encoded=0
input_data = pd.DataFrame([[
        loan_amnt, int_rate, monthly_payment, dti, delinq_2yrs,
        open_acc, pub_rec, revol_bal, revol_util, total_acc,
        fico_score, emp_length,
        term_encoded, grade_encoded, home_ownership_encoded,
        0, purpose_encoded
    ]], columns=features)
print('Input shape', input_data.shape)
input_scaled = scaler.transform(input_data)
proba = model.predict_proba(input_scaled)[0][1]
print('proba', proba)
