import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Load Model & Scaler ---
try:
    with open("Churn_Prediction_Best_Model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_path.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Get features directly from scaler to ensure perfect matching
    if hasattr(scaler, 'feature_names_in_'):
        MODEL_FEATURES = list(scaler.feature_names_in_)
    else:
        # Fallback manual list (from your log)
        MODEL_FEATURES = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'gender_gender', 'Partner_Partner', 'Dependents_Dependents',
            'PhoneService_PhoneService',
            'MultipleLines_MultipleLines_No phone service', 'MultipleLines_MultipleLines_Yes',
            'InternetService_InternetService_Fiber optic', 'InternetService_InternetService_No',
            'OnlineSecurity_OnlineSecurity_No internet service', 'OnlineSecurity_OnlineSecurity_Yes',
            'OnlineBackup_OnlineBackup_No internet service', 'OnlineBackup_OnlineBackup_Yes',
            'DeviceProtection_DeviceProtection_No internet service', 'DeviceProtection_DeviceProtection_Yes',
            'TechSupport_TechSupport_No internet service', 'TechSupport_TechSupport_Yes',
            'StreamingTV_StreamingTV_No internet service', 'StreamingTV_StreamingTV_Yes',
            'StreamingMovies_StreamingMovies_No internet service', 'StreamingMovies_StreamingMovies_Yes',
            'Contract_Contract_One year', 'Contract_Contract_Two year',
            'PaperlessBilling_PaperlessBilling',
            'PaymentMethod_PaymentMethod_Credit card (automatic)',
            'PaymentMethod_PaymentMethod_Electronic check', 'PaymentMethod_PaymentMethod_Mailed check',
            'SIM_SIM_BSNL', 'SIM_SIM_Jio', 'SIM_SIM_Vi',
            'DeviceType_DeviceType',
            'Region_Region_Sub Urban', 'Region_Region_Urban'
        ]

    print(f"✅ Model loaded successfully. Expecting {len(MODEL_FEATURES)} features.")

except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 1. Initialize DataFrame with 0s (Default state)
        input_df = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)

        # 2. Extract Basic Data
        tenure = float(data.get('tenure', 0))
        monthly_charges = float(data.get('MonthlyCharges', 0))

        # --- FIX: Auto-calculate TotalCharges if missing or 0 ---
        # The model fails if TotalCharges is 0 but tenure is high
        total_charges = data.get('TotalCharges', '')
        if total_charges == '' or float(total_charges) == 0:
            total_charges = tenure * monthly_charges
        else:
            total_charges = float(total_charges)

        # Set Numerical Columns
        input_df['SeniorCitizen'] = int(data.get('SeniorCitizen', 0))
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = monthly_charges
        input_df['TotalCharges'] = total_charges

        # 3. Categorical Mappings (Matching your specific feature names)

        # Gender (Assuming 1=Male based on your 'gender_gender')
        if 'gender_gender' in MODEL_FEATURES:
            input_df['gender_gender'] = 1 if data.get('gender') == 'Male' else 0

        # Partner
        if 'Partner_Partner' in MODEL_FEATURES:
            input_df['Partner_Partner'] = 1 if data.get('Partner') == 'Yes' else 0

        # Dependents
        if 'Dependents_Dependents' in MODEL_FEATURES:
            input_df['Dependents_Dependents'] = 1 if data.get('Dependents') == 'Yes' else 0

        # Phone Service
        if 'PhoneService_PhoneService' in MODEL_FEATURES:
            input_df['PhoneService_PhoneService'] = 1 if data.get('PhoneService') == 'Yes' else 0

        # Paperless Billing
        if 'PaperlessBilling_PaperlessBilling' in MODEL_FEATURES:
            input_df['PaperlessBilling_PaperlessBilling'] = 1 if data.get('PaperlessBilling') == 'Yes' else 0

        # Device Type (New Device = 1)
        if 'DeviceType_DeviceType' in MODEL_FEATURES:
            input_df['DeviceType_DeviceType'] = 1 if data.get('DeviceType') == 'New Device' else 0

        # 4. One-Hot Encoding Helper
        # This function sets the column to 1 if it exists in the model features
        def set_column(col_name):
            if col_name in MODEL_FEATURES:
                input_df[col_name] = 1

        # MultipleLines
        if data.get('MultipleLines') == 'No phone service':
            set_column('MultipleLines_MultipleLines_No phone service')
        elif data.get('MultipleLines') == 'Yes':
            set_column('MultipleLines_MultipleLines_Yes')

        # InternetService
        if data.get('InternetService') == 'Fiber optic':
            set_column('InternetService_InternetService_Fiber optic')
        elif data.get('InternetService') == 'No':
            set_column('InternetService_InternetService_No')

        # Online Services
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']

        for svc in services:
            val = data.get(svc)
            if val == 'Yes':
                set_column(f"{svc}_{svc}_Yes")
            elif val == 'No internet service':
                set_column(f"{svc}_{svc}_No internet service")

        # Contract
        contract = data.get('Contract')
        if contract == 'One year':
            set_column('Contract_Contract_One year')
        elif contract == 'Two year':
            set_column('Contract_Contract_Two year')

        # Payment Method
        pay = data.get('PaymentMethod')
        if pay == 'Credit card (automatic)':
            set_column('PaymentMethod_PaymentMethod_Credit card (automatic)')
        elif pay == 'Electronic check':
            set_column('PaymentMethod_PaymentMethod_Electronic check')
        elif pay == 'Mailed check':
            set_column('PaymentMethod_PaymentMethod_Mailed check')

        # SIM
        sim = data.get('SIM')
        if sim == 'BSNL':
            set_column('SIM_SIM_BSNL')
        elif sim == 'Jio':
            set_column('SIM_SIM_Jio')
        elif sim == 'Vi':
            set_column('SIM_SIM_Vi')

        # Region
        region = data.get('Region')
        if region == 'Sub Urban':
            set_column('Region_Region_Sub Urban')
        elif region == 'Urban':
            set_column('Region_Region_Urban')

        # --- PREDICTION ---
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)

        # Log to terminal for debugging
        print(f"\n--- DEBUG INFO ---")
        print(f"Tenure: {tenure}, Monthly: {monthly_charges}, Total: {total_charges}")
        print(f"Raw Probabilities: {probabilities[0]}")

        # Standard Sklearn: Class 0 is 'No', Class 1 is 'Yes'
        churn_prob = probabilities[0][1] * 100

        result = "Yes" if churn_prob > 50 else "No"

        return jsonify({
            'churn': result,
            'probability': round(churn_prob, 2)
        })

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)