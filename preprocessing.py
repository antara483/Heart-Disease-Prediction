
import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess heart disease data"""
    df = df.copy()
    
    # Map binary columns
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    if 'fasting_blood_sugar' in df.columns:
        df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
            'Lower than 120 mg/ml': 0,
            'Greater than 120 mg/ml': 1
        })
    
    if 'exercise_induced_angina' in df.columns:
        df['exercise_induced_angina'] = df['exercise_induced_angina'].map({'No': 0, 'Yes': 1})
    
    # Handle categorical columns
    categorical_cols = ['chest_pain_type', 'rest_ecg', 'slope', 'thalassemia', 'vessels_colored_by_flourosopy']
    
    for col in categorical_cols:
        if col in df.columns:
            # Convert to categorical codes
            df[col] = df[col].astype('category').cat.codes
            # Replace -1 (unknown) with NaN
            df[col] = df[col].replace(-1, np.nan)
    
    # Numeric clipping
    if 'resting_blood_pressure' in df.columns:
        df['resting_blood_pressure'] = df['resting_blood_pressure'].clip(upper=200)
    
    if 'cholestoral' in df.columns:
        df['cholestoral'] = df['cholestoral'].clip(upper=400)
    
    if 'oldpeak' in df.columns:
        df['oldpeak'] = df['oldpeak'].clip(upper=5.0)
    
    return df
