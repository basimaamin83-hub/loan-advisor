import streamlit as st
import joblib
import os
import traceback

st.set_page_config(page_title="Debug", layout="wide")
st.title("🔍 Debug Mode")

try:
    st.write("📂 Current directory:", os.getcwd())
    st.write("📂 Files in current directory:", os.listdir('.'))
    
    if os.path.exists('models'):
        st.write("✅ models folder exists")
        st.write("📄 Files in models:", os.listdir('models'))
        
        # Try loading each file
        model = joblib.load('models/loan_model.pkl')
        st.success("✅ Model loaded successfully!")
        st.write(f"Model type: {type(model)}")
        
        scaler = joblib.load('models/scaler.pkl')
        st.success("✅ Scaler loaded successfully!")
        
        features = joblib.load('models/features.pkl')
        st.success("✅ Features loaded successfully!")
        st.write(f"Number of features: {len(features)}")
        st.write(f"Features: {features}")
        
        label_encoders = joblib.load('models/label_encoders.pkl')
        st.success("✅ Label encoders loaded successfully!")
        st.write(f"Encoders: {list(label_encoders.keys())}")
        
        st.success("🎉 All models loaded successfully! Ready to run the main app.")
    else:
        st.error("❌ models folder not found!")
        
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.code(traceback.format_exc())
