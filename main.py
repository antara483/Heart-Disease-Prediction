


from datetime import datetime
from fastapi.responses import StreamingResponse
# from pdf_generator import PDFReportGenerator
from fastapi import FastAPI, Form, Request,WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import json
templates = Jinja2Templates(directory="templates")
# Load trained model and optimal threshold
model_data = joblib.load("heart_disease_model_completed_finally.pkl")
model = model_data['model']
OPTIMAL_THRESHOLD = model_data['optimal_threshold']

app = FastAPI(
    title="❤️ Heart Disease Prediction API",
    description="Predict heart disease based on patient health data.",
    version="2.0"
)

templates = Jinja2Templates(directory="templates")



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Map binary categorical values with proper error handling
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    # Handle any unmapped values
    df['sex'] = df['sex'].fillna(0)  # Default to Female if invalid
    
    df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
        'Lower than 120 mg/ml': 0,
        'Greater than 120 mg/ml': 1
    })
    df['fasting_blood_sugar'] = df['fasting_blood_sugar'].fillna(0)  # Default to normal
    
    df['exercise_induced_angina'] = df['exercise_induced_angina'].map({'No': 0, 'Yes': 1})
    df['exercise_induced_angina'] = df['exercise_induced_angina'].fillna(0)  # Default to No

    # Fix vessels_colored_by_flourosopy with proper handling
    vessel_map = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 0}  # Map Four to 0 instead of NaN
    df['vessels_colored_by_flourosopy'] = df['vessels_colored_by_flourosopy'].map(vessel_map)
    df['vessels_colored_by_flourosopy'] = df['vessels_colored_by_flourosopy'].fillna(0)  # Default to Zero

    # Fix thalassemia - use explicit mapping instead of replacement
    thalassemia_map = {
        'normal': 3,
        'fixed_defect': 6, 
        'reversible_defect': 7,
        'No': 3  # Map 'No' to normal
    }
    df['thalassemia'] = df['thalassemia'].map(thalassemia_map)
    df['thalassemia'] = df['thalassemia'].fillna(3)  # Default to normal

    # Real-world capping
    df['resting_blood_pressure'] = df['resting_blood_pressure'].clip(upper=200)
    df['cholestoral'] = df['cholestoral'].clip(upper=400)
    df['oldpeak'] = df['oldpeak'].clip(upper=5.0)

    # Convert categorical columns with proper mapping
    chest_pain_map = {
        'typical_angina': 1,
        'atypical_angina': 2,
        'non_anginal_pain': 3,
        'asymptomatic': 4
    }
    
    rest_ecg_map = {
        'normal': 0,
        'st_t_wave_abnormality': 1,
        'left_ventricular_hypertrophy': 2
    }
    
    slope_map = {
        'upsloping': 1,
        'flat': 2,
        'downsloping': 3
    }

    # Apply mappings with defaults
    if 'chest_pain_type' in df.columns:
        df['chest_pain_type'] = df['chest_pain_type'].map(chest_pain_map).fillna(4)  # Default to asymptomatic
    
    if 'rest_ecg' in df.columns:
        df['rest_ecg'] = df['rest_ecg'].map(rest_ecg_map).fillna(0)  # Default to normal
    
    if 'slope' in df.columns:
        df['slope'] = df['slope'].map(slope_map).fillna(1)  # Default to upsloping

    return df

# ====== EVIDENCE-BASED RISK ADJUSTMENT ======
# def evidence_based_risk_adjustment(input_data, raw_probability):
#     age = float(input_data['age'].iloc[0])
#     bp = float(input_data['resting_blood_pressure'].iloc[0])
#     cholesterol = float(input_data['cholestoral'].iloc[0])
#     max_hr = float(input_data['Max_heart_rate'].iloc[0])
#     oldpeak = float(input_data['oldpeak'].iloc[0])
#     exang = int(input_data['exercise_induced_angina'].iloc[0])
#     sex = int(input_data['sex'].iloc[0])
#     fbs = int(input_data['fasting_blood_sugar'].iloc[0])

#     adjustment_multiplier = 1.0
#     adjustment_factors = []

#     # Age
#     if age < 35:
#         adjustment_multiplier *= 0.2
#         adjustment_factors.append("Very young age (<35): 80% risk reduction")
#     elif age < 45:
#         adjustment_multiplier *= 0.4
#         adjustment_factors.append("Young age (35-44): 60% risk reduction")
#     elif age < 55:
#         adjustment_multiplier *= 0.7
#         adjustment_factors.append("Middle age (45-54): 30% risk reduction")

#     # Blood pressure
#     if bp < 120:
#         adjustment_multiplier *= 0.6
#         adjustment_factors.append("Optimal BP (<120 mmHg): 40% risk reduction")
#     elif bp < 130:
#         adjustment_multiplier *= 0.8
#         adjustment_factors.append("Normal BP (120-129 mmHg): 20% risk reduction")
#     if bp > 180:
#         adjustment_multiplier *= 2.0
#         adjustment_factors.append("Stage 3 Hypertension (>180 mmHg): 100% risk increase")

#     # Cholesterol
#     if cholesterol < 200:
#         adjustment_multiplier *= 0.7
#         adjustment_factors.append("Desirable cholesterol (<200 mg/dL): 30% risk reduction")
#     if cholesterol > 240:
#         adjustment_multiplier *= 1.5
#         adjustment_factors.append("High cholesterol (>240 mg/dL): 50% risk increase")

#     # Exercise capacity
#     if max_hr > 140 and exang == 0:
#         adjustment_multiplier *= 0.5
#         adjustment_factors.append("Good exercise capacity, no angina: 50% risk reduction")
#     if exang == 1:
#         adjustment_multiplier *= 2.0
#         adjustment_factors.append("Exercise-induced angina: 100% risk increase")

#     # ST depression (oldpeak)
#     if oldpeak < 0.5:
#         adjustment_multiplier *= 0.7
#         adjustment_factors.append("Minimal ST depression (<0.5mm): 30% risk reduction")
#     if oldpeak > 2.0:
#         adjustment_multiplier *= 2.0
#         adjustment_factors.append("Significant ST depression (>2.0mm): 100% risk increase")

#     # Gender
#     if sex == 0 and age < 55:
#         adjustment_multiplier *= 0.6
#         adjustment_factors.append("Pre-menopausal female: 40% risk reduction")

#     # Blood sugar
#     if fbs == 0:
#         adjustment_multiplier *= 0.8
#         adjustment_factors.append("Normal fasting glucose: 20% risk reduction")

#     # Compute final adjusted probability
#     adjusted_prob = raw_probability * adjustment_multiplier
#     adjusted_prob = max(0.01, min(adjusted_prob, 0.95))  # bounds
#     significant_adjustment = abs(raw_probability - adjusted_prob) > 0.2

#     return {
#         'adjusted_probability': adjusted_prob,
#         'adjustment_factors': adjustment_factors,
#         'adjustment_multiplier': adjustment_multiplier,
#         'significant_adjustment': significant_adjustment,
#         'final_prediction': 1 if adjusted_prob > 0.5 else 0
#     }

def evidence_based_risk_adjustment(input_data, raw_probability):
    # Safely extract values with NaN handling
    def safe_extract(column, default=0):
        try:
            value = input_data[column].iloc[0]
            if pd.isna(value):
                return default
            return float(value)
        except (KeyError, IndexError, ValueError):
            return default

    age = safe_extract('age')
    bp = safe_extract('resting_blood_pressure')
    cholesterol = safe_extract('cholestoral')
    max_hr = safe_extract('Max_heart_rate')
    oldpeak = safe_extract('oldpeak')
    
    # For binary fields, use integer conversion with NaN handling
    try:
        exang_value = input_data['exercise_induced_angina'].iloc[0]
        exang = 0 if pd.isna(exang_value) else int(float(exang_value))
    except (KeyError, IndexError, ValueError):
        exang = 0

    try:
        sex_value = input_data['sex'].iloc[0]
        sex = 0 if pd.isna(sex_value) else int(float(sex_value))
    except (KeyError, IndexError, ValueError):
        sex = 0

    try:
        fbs_value = input_data['fasting_blood_sugar'].iloc[0]
        fbs = 0 if pd.isna(fbs_value) else int(float(fbs_value))
    except (KeyError, IndexError, ValueError):
        fbs = 0

    adjustment_multiplier = 1.0
    adjustment_factors = []

    # Age adjustments
    if age < 35:
        adjustment_multiplier *= 0.2
        adjustment_factors.append("Very young age (<35): 80% risk reduction")
    elif age < 45:
        adjustment_multiplier *= 0.4
        adjustment_factors.append("Young age (35-44): 60% risk reduction")
    elif age < 55:
        adjustment_multiplier *= 0.7
        adjustment_factors.append("Middle age (45-54): 30% risk reduction")

    # Blood pressure adjustments
    if bp < 120:
        adjustment_multiplier *= 0.6
        adjustment_factors.append("Optimal BP (<120 mmHg): 40% risk reduction")
    elif bp < 130:
        adjustment_multiplier *= 0.8
        adjustment_factors.append("Normal BP (120-129 mmHg): 20% risk reduction")
    if bp > 180:
        adjustment_multiplier *= 2.0
        adjustment_factors.append("Stage 3 Hypertension (>180 mmHg): 100% risk increase")

    # Cholesterol adjustments
    if cholesterol < 200:
        adjustment_multiplier *= 0.7
        adjustment_factors.append("Desirable cholesterol (<200 mg/dL): 30% risk reduction")
    if cholesterol > 240:
        adjustment_multiplier *= 1.5
        adjustment_factors.append("High cholesterol (>240 mg/dL): 50% risk increase")

    # Exercise capacity adjustments
    if max_hr > 140 and exang == 0:
        adjustment_multiplier *= 0.5
        adjustment_factors.append("Good exercise capacity, no angina: 50% risk reduction")
    if exang == 1:
        adjustment_multiplier *= 2.0
        adjustment_factors.append("Exercise-induced angina: 100% risk increase")

    # ST depression (oldpeak) adjustments
    if oldpeak < 0.5:
        adjustment_multiplier *= 0.7
        adjustment_factors.append("Minimal ST depression (<0.5mm): 30% risk reduction")
    if oldpeak > 2.0:
        adjustment_multiplier *= 2.0
        adjustment_factors.append("Significant ST depression (>2.0mm): 100% risk increase")

    # Gender adjustments
    if sex == 0 and age < 55:
        adjustment_multiplier *= 0.6
        adjustment_factors.append("Pre-menopausal female: 40% risk reduction")

    # Blood sugar adjustments
    if fbs == 0:
        adjustment_multiplier *= 0.8
        adjustment_factors.append("Normal fasting glucose: 20% risk reduction")

    # Compute final adjusted probability
    adjusted_prob = raw_probability * adjustment_multiplier
    adjusted_prob = max(0.01, min(adjusted_prob, 0.95))  # bounds
    significant_adjustment = abs(raw_probability - adjusted_prob) > 0.2

    return {
        'adjusted_probability': adjusted_prob,
        'adjustment_factors': adjustment_factors,
        'adjustment_multiplier': adjustment_multiplier,
        'significant_adjustment': significant_adjustment,
        'final_prediction': 1 if adjusted_prob > 0.5 else 0
    }
# ====== HELPER FUNCTIONS ======
def get_risk_level(probability):
    if probability < 0.05:
        return "Very Low Risk"
    elif probability < 0.1:
        return "Low Risk"
    elif probability < 0.2:
        return "Borderline Risk"
    elif probability < 0.4:
        return "Intermediate Risk"
    else:
        return "High Risk"

def get_clinical_message(adjustment_result):
    prob = adjustment_result['adjusted_probability']
    if prob < 0.05:
        return "✅ Very low probability of heart disease. Maintain healthy lifestyle."
    elif prob < 0.1:
        return "✅ Low probability of heart disease. Routine follow-up recommended."
    elif prob < 0.2:
        return "🟡 Borderline risk. Consider cardiovascular risk assessment."
    elif prob < 0.4:
        return "🟠 Intermediate risk. Further evaluation may be beneficial."
    else:
        return "🔴 High probability of heart disease. Medical evaluation recommended."
def validate_input_data(input_dict):
    """Validate input data before processing"""
    required_fields = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
    
    for field in required_fields:
        if field not in input_dict or input_dict[field] is None:
            raise ValueError(f"Missing required field: {field}")
        
        value = input_dict[field]
        if isinstance(value, (int, float)) and (pd.isna(value) or np.isinf(value)):
            raise ValueError(f"Invalid value for {field}: {value}")
    
    return True
# class RealTimeCalculator:
#     def __init__(self):
#         self.risk_factors = {}
    
#     def calculate_heart_age(self, input_data):
#         """Calculate heart age based on risk factors"""
#         chronological_age = input_data['age']
#         heart_age = chronological_age
        
#         # Risk adjustments
#         if input_data['resting_blood_pressure'] > 140:
#             heart_age += 5
#         if input_data['cholestoral'] > 240:
#             heart_age += 4
#         if input_data['exercise_induced_angina'] == 1:
#             heart_age += 7
#         if input_data['oldpeak'] > 2.0:
#             heart_age += 3
            
#         return max(chronological_age, heart_age)
    
#     def calculate_risk_score(self, input_data):
#         """Calculate Framingham-like risk score"""
#         score = 0
        
#         # Age
#         age = input_data['age']
#         if age >= 60: score += 8
#         elif age >= 50: score += 6
#         elif age >= 40: score += 4
        
#         # Blood Pressure
#         bp = input_data['resting_blood_pressure']
#         if bp >= 160: score += 5
#         elif bp >= 140: score += 3
#         elif bp >= 130: score += 1
        
#         # Cholesterol
#         chol = input_data['cholestoral']
#         if chol >= 240: score += 4
#         elif chol >= 200: score += 2
        
#         # Other factors
#         if input_data['exercise_induced_angina'] == 1: score += 3
#         if input_data['oldpeak'] > 2.0: score += 2
#         if input_data['fasting_blood_sugar'] == 1: score += 2
        
#         return min(score, 20)  # Max score 20

#         # Add to main.py


# ====== REAL-TIME CALCULATOR ======
class RealTimeCalculator:
    def __init__(self):
        self.risk_factors = {}
    
    def calculate_heart_age(self, input_data):
        """Calculate heart age based on risk factors"""
        try:
            chronological_age = float(input_data.get('age', 50))
            heart_age = chronological_age
            
            # Risk adjustments
            bp = float(input_data.get('resting_blood_pressure', 120))
            chol = float(input_data.get('cholestoral', 200))
            oldpeak = float(input_data.get('oldpeak', 0))
            exang = int(input_data.get('exercise_induced_angina', 0))
            
            if bp > 140:
                heart_age += 5
            elif bp > 130:
                heart_age += 3
                
            if chol > 240:
                heart_age += 4
            elif chol > 200:
                heart_age += 2
                
            if exang == 1:
                heart_age += 7
                
            if oldpeak > 2.0:
                heart_age += 3
            elif oldpeak > 1.0:
                heart_age += 1
                
            return int(max(chronological_age, heart_age))
        except:
            return chronological_age
    
    def calculate_risk_score(self, input_data):
        """Calculate Framingham-like risk score"""
        try:
            score = 0
            
            # Age
            age = float(input_data.get('age', 50))
            if age >= 60: score += 8
            elif age >= 50: score += 6
            elif age >= 40: score += 4
            
            # Blood Pressure
            bp = float(input_data.get('resting_blood_pressure', 120))
            if bp >= 160: score += 5
            elif bp >= 140: score += 3
            elif bp >= 130: score += 1
            
            # Cholesterol
            chol = float(input_data.get('cholestoral', 200))
            if chol >= 240: score += 4
            elif chol >= 200: score += 2
            
            # Other factors
            if int(input_data.get('exercise_induced_angina', 0)) == 1: 
                score += 3
            if float(input_data.get('oldpeak', 0)) > 2.0: 
                score += 2
            if int(input_data.get('fasting_blood_sugar', 0)) == 1: 
                score += 2
            
            return min(score, 20)
        except:
            return 0

# ====== EDUCATION CONTENT ======
class EducationContent:
    def __init__(self):
        self.content_db = {
            'blood_pressure': {
                'title': 'Understanding Blood Pressure',
                'content': 'Blood Pressure Guidelines',
                'tips': [
                    'Reduce sodium intake to less than 2,300 mg daily',
                    'Exercise regularly - aim for 150 minutes per week',
                    'Maintain healthy weight',
                    'Limit alcohol consumption'
                ]
            },
            'cholesterol': {
                'title': 'Cholesterol Management',
                'content': 'Cholesterol Levels',
                'tips': [
                    'Choose foods low in saturated and trans fats',
                    'Eat more soluble fiber (oats, beans, apples)',
                    'Include omega-3 fatty acids (fish, walnuts)',
                    'Exercise regularly to raise HDL (good cholesterol)'
                ]
            },
            'heart_health': {
                'title': 'General Heart Health',
                'content': 'Maintaining a Healthy Heart',
                'tips': [
                    'Aim for 7-9 hours of quality sleep nightly',
                    'Manage stress through meditation or yoga',
                    'Eat a balanced diet rich in fruits and vegetables',
                    'Avoid tobacco products'
                ]
            }
        }
    
    def get_personalized_advice(self, patient_data):
        advice = []
        
        try:
            bp = float(patient_data.get('resting_blood_pressure', 120))
            chol = float(patient_data.get('cholestoral', 200))
            age = float(patient_data.get('age', 50))
            
            if bp > 130:
                advice.append(self.content_db['blood_pressure'])
            
            if chol > 200:
                advice.append(self.content_db['cholesterol'])
            
            # Always include general heart health advice
            advice.append(self.content_db['heart_health'])
            
        except:
            # Fallback content if there's any error
            advice.append(self.content_db['heart_health'])
        
        return advice

# Initialize components
calculator = RealTimeCalculator()
education_content = EducationContent()



@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
def form(request: Request):
    numeric_ranges = {
        "age": (18, 100),
        "resting_blood_pressure": (90, 200),
        "cholestoral": (120, 600),
        "Max_heart_rate": (70, 210),
        "oldpeak": (0, 6.5)
    }
    return templates.TemplateResponse("form.html", {"request": request, "ranges": numeric_ranges})



@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    patient_name: str = Form(...),
    patient_id: str = Form(None),
    age: float = Form(...),
    sex: str = Form(...),
    resting_blood_pressure: float = Form(...),
    cholestoral: float = Form(...),
    Max_heart_rate: float = Form(...),
    oldpeak: float = Form(...),
    fasting_blood_sugar: str = Form(...),
    exercise_induced_angina: str = Form(...),
    chest_pain_type: str = Form(...),
    rest_ecg: str = Form(...),
    slope: str = Form(...),
    thalassemia: str = Form(...),
    vessels_colored_by_flourosopy: str = Form(...)
):
    # Store form data for display
    form_data = {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "age": age, "sex": sex, "resting_blood_pressure": resting_blood_pressure,
        "cholestoral": cholestoral, "Max_heart_rate": Max_heart_rate, "oldpeak": oldpeak,
        "fasting_blood_sugar": fasting_blood_sugar, "exercise_induced_angina": exercise_induced_angina,
        "chest_pain_type": chest_pain_type, "rest_ecg": rest_ecg, "slope": slope,
        "thalassemia": thalassemia, "vessels_colored_by_flourosopy": vessels_colored_by_flourosopy
    }
    
    try:
        input_df = pd.DataFrame([form_data])

        print("=== RAW INPUT DATA ===")
        print(input_df)

        input_df_cleaned = clean_data(input_df)
        
        print("=== CLEANED DATA ===")
        print(input_df_cleaned)
        print("Missing values after cleaning:")
        print(input_df_cleaned.isnull().sum())

        if model is None:
            raise Exception("Model not loaded - please check if the model file exists")

        # Check if we have the required features
        missing_features = set(model.feature_names_in_) - set(input_df_cleaned.columns)
        if missing_features:
            raise Exception(f"Missing features in data: {missing_features}")

        # Handle any remaining missing values
        input_df_cleaned = input_df_cleaned[model.feature_names_in_]
        
        # Fill any remaining NaN values with defaults
        input_df_cleaned = input_df_cleaned.fillna({
            'sex': 0,
            'fasting_blood_sugar': 0,
            'exercise_induced_angina': 0,
            'vessels_colored_by_flourosopy': 0,
            'thalassemia': 3,
            'chest_pain_type': 4,
            'rest_ecg': 0,
            'slope': 1
        })

        print("=== FINAL DATA FOR PREDICTION ===")
        print(input_df_cleaned)
        print("Any remaining missing values:", input_df_cleaned.isnull().sum().sum())

        probability = model.predict_proba(input_df_cleaned)[0][1]
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0

        # Continue with your existing result processing...
        adjustment_result = evidence_based_risk_adjustment(input_df_cleaned, probability)
        
        # Calculate real-time metrics with fallbacks
        try:
            heart_age = calculator.calculate_heart_age(form_data)
        except:
            heart_age = form_data['age']  # Default to actual age if calculation fails
            
        try:
            risk_score = calculator.calculate_risk_score(form_data)
        except:
            risk_score = 0  # Default to 0 if calculation fails
        
        # Get personalized education content
        try:
            personalized_advice = education_content.get_personalized_advice(form_data)
        except:
            # Fallback education content
            personalized_advice = [{
                'title': 'Heart Health Tips',
                'content': 'General cardiovascular health recommendations',
                'tips': [
                    'Maintain a balanced diet rich in fruits and vegetables',
                    'Exercise regularly for at least 30 minutes daily',
                    'Get regular health check-ups',
                    'Avoid smoking and limit alcohol consumption'
                ]
            }]

        # Determine risk level for display
        adjusted_prob = adjustment_result['adjusted_probability']
        if adjusted_prob < 0.05:
            risk_level = "Very Low Risk"
        elif adjusted_prob < 0.1:
            risk_level = "Low Risk"
        elif adjusted_prob < 0.2:
            risk_level = "Borderline Risk"
        elif adjusted_prob < 0.4:
            risk_level = "Intermediate Risk"
        else:
            risk_level = "High Risk"

        result = {
            "request": request,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "prediction": "Yes" if adjustment_result['final_prediction'] == 1 else "No",
            "probability": f"{adjustment_result['adjusted_probability'] * 100:.1f}%",
            "risk_level": risk_level,
            "message": get_clinical_message(adjustment_result),
            "adjustment_applied": adjustment_result['significant_adjustment'],
            "factors_considered": adjustment_result['adjustment_factors'],
            "heart_age": heart_age,
            "risk_score": risk_score,
            "personalized_advice": personalized_advice,
            "form_data": form_data,
            "error": None,
            "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "report_id": f"HR{datetime.now().strftime('%Y%m%d%H%M%S')}",
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # Return results page with error information
        result = {
            "request": request,
            "prediction": "Error",
            "probability": "0.0%",
            "risk_level": "Unable to assess",
            "message": f"❌ Assessment incomplete due to error: {str(e)}",
            "adjustment_applied": False,
            "factors_considered": [],
            "heart_age": form_data['age'],  # Default to actual age
            "risk_score": 0,
            "personalized_advice": [{
                'title': 'Heart Health Tips',
                'content': 'General cardiovascular health recommendations',
                'tips': [
                    'Maintain a balanced diet rich in fruits and vegetables',
                    'Exercise regularly for at least 30 minutes daily',
                    'Get regular health check-ups',
                    'Avoid smoking and limit alcohol consumption'
                ]
            }],
            "form_data": form_data,
            "error": str(e)  # Include error message
        }
    
    return templates.TemplateResponse("result.html", result)
@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    # This is just sample data for demonstration
    # In a real scenario, you'd get this from the prediction or session
    form_data = {
        "age": 52,
        "resting_blood_pressure": 125,
        "cholestoral": 212,
        "Max_heart_rate": 168,
        "oldpeak": 1.0,
        "exercise_induced_angina": "No",
        "sex": "Male",
        "fasting_blood_sugar": "Lower than 120 mg/ml",
        "chest_pain_type": "typical_angina",
        "rest_ecg": "normal", 
        "slope": "upsloping",
        "thalassemia": "normal",
        "vessels_colored_by_flourosopy": "Zero"
    }
    
    heart_age = 55
    risk_score = 7
    risk_level = "Medium"
    factors_considered = ["High Cholesterol", "High Blood Pressure"]
    message = "Your cardiovascular health is moderate."
    personalized_advice = [
        {"title": "Heart Health", "content": "Maintain regular exercise.", "tips": ["Walk 30 min daily", "Avoid smoking"]}
    ]

    return templates.TemplateResponse("results.html", {
        "request": request,
        "form_data": form_data,  # This was missing
        "heart_age": heart_age,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "factors_considered": factors_considered,
        "message": message,
        "personalized_advice": personalized_advice
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
