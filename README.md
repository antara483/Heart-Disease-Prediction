# ❤️Heart Disease Prediction

A machine learning project for predicting the likelihood of heart disease based on patient medical attributes and lifestyle factors.

Overview
Here’s how the Home Page looks:

![Home Page](images/Home%20Page.png)

And here’s a very low risk report:

![Very low Risk Report](images/A%20Very%20low%20risk%20Report.png)


## 🚀 Features
🤖 Machine Learning: Ensemble model (Random Forest + Logistic Regression) with 80.3% accuracy

🏥 Clinical Safety: 91% reduction in false negatives for patient safety

⚡ Real-time Predictions: FastAPI backend with instant risk assessment

📊 Evidence-Based Adjustments: Medical guideline-informed risk modifications

📄 PDF Reports: Professional medical report generation

🎨 Beautiful UI: Responsive Tailwind CSS interface

🔧 Threshold Optimization: Custom threshold tuning for medical safety


## 🛠️ Technology Stack
### Backend
FastAPI - Modern Python web framework

Scikit-learn - Machine learning algorithms

Pandas & NumPy - Data processing

Joblib - Model serialization

### Frontend
HTML5 - Page structure

Tailwind CSS - Styling and responsive design

JavaScript - Real-time risk calculator

Font Awesome - Icons

Machine Learning
Ensemble Methods - Random Forest + Logistic Regression voting

Feature Engineering - Clinical feature optimization

Threshold Optimization - Medical safety-focused tuning

# 🏥 Understanding the Input Fields
## 📋 Patient Basic Information
#### 1. Age
What it measures: The patient's age in years

Why it matters: Heart disease risk increases with age as arteries naturally stiffen and plaque accumulates over time.
Clinical ranges:

Under 45: Very low baseline risk

45-65: Gradual risk increase

Over 65: Highest risk category

#### Medical insight: Regular heart health screening becomes more important after age 45.

### 2. Sex
What it measures: Biological sex (Male/Female)

Why it matters: Men generally have higher baseline heart disease risk, especially before women reach menopause.
Clinical significance:

Pre-menopausal women have some natural protection due to estrogen

Post-menopausal women's risk becomes similar to men's

Men should start monitoring earlier in life

### 3. Resting Blood Pressure
What it measures: Blood pressure in millimeters of mercury (mmHg) while at rest

Why it matters: High blood pressure damages artery walls over time, forcing the heart to work harder.
Clinical categories:

Optimal: Below 120 mmHg

Normal: 120-129 mmHg

High: 130-139 mmHg

Very High: 140+ mmHg

#### Medical insight: Often called the "silent killer" because it may show no symptoms until significant damage occurs.

### 4. Cholesterol
What it measures: Total serum cholesterol in milligrams per deciliter (mg/dL)

Why it matters: High cholesterol leads to plaque buildup in arteries, restricting blood flow to the heart.
Clinical ranges:

Desirable: Below 200 mg/dL

Borderline High: 200-239 mg/dL

High: 240+ mg/dL

#### Medical insight: Cholesterol management is crucial for long-term heart health.

## 🏃 Clinical Measurements & Exercise Capacity
### 5. Maximum Heart Rate
What it measures: Highest heart rate achieved during exercise (beats per minute)

Why it matters: Indicates cardiovascular fitness and exercise capacity.
Clinical interpretation:

Excellent: 140-180 BPM (good fitness)

Average: 120-140 BPM

Poor: Below 120 BPM (may indicate underlying issues)

#### Medical insight: Rough estimate = 220 minus age. Higher values suggest better heart function.

### 6. Oldpeak (ST Depression)
What it measures: ST segment depression on ECG during exercise (in millimeters)

Why it matters: Measures stress-induced changes in heart electrical activity.
Clinical significance:

Normal: 0-1.0 mm (minimal stress response)

Mild: 1.0-2.0 mm (borderline concern)

Significant: 2.0+ mm (possible ischemia)

#### Medical insight: Higher values may indicate reduced blood flow to heart muscle during stress.

### 7. Fasting Blood Sugar
What it measures: Blood glucose levels after fasting

Why it matters: High levels may indicate diabetes or pre-diabetes, major heart disease risk factors.
Clinical ranges:

Normal: Lower than 120 mg/ml

High: Greater than 120 mg/ml

#### Medical insight: Diabetes significantly accelerates heart disease progression.

## 💓 Symptoms & Diagnostic Test Results

### 8. Exercise-Induced Angina
What it measures: Whether chest pain occurs during physical activity

Why it matters: Chest pain during exertion strongly suggests coronary artery disease.
Clinical significance:

No: Lower risk profile

Yes: Significant concern requiring medical evaluation

#### Medical insight: One of the most direct indicators of heart blood flow problems.

### 9. Chest Pain Type
What it measures: Nature and characteristics of chest discomfort

Why it matters: Different pain patterns suggest different types of heart issues.
Clinical progression (increasing concern):

Typical Angina: Predictable chest pain during exertion

Atypical Angina: Less predictable chest discomfort

Non-anginal Pain: Chest pain not clearly heart-related

Asymptomatic: No pain but other risk factors present (most concerning)

#### Medical insight: "Silent" heart disease without pain can be particularly dangerous.

### 10. Resting ECG Results
What it measures: Electrical activity of the heart at rest

Why it matters: Identifies underlying heart rhythm and structural issues.
Clinical progression (increasing concern):

Normal: No significant abnormalities

ST-T Wave Abnormality: Possible reduced blood flow

Left Ventricular Hypertrophy: Thickened heart muscle (often from high BP)

#### Medical insight: ECG changes can reveal "silent" heart damage.

### 11. Slope of ST Segment
What it measures: Pattern of ST segment during peak exercise

Why it matters: Indicates how the heart responds to stress.
Clinical progression (increasing concern):

Upsloping: Normal stress response

Flat: Possible reduced blood flow

Downsloping: Strong indicator of ischemia (reduced oxygen)

#### Medical insight: The pattern change during exercise is more important than resting values.

### 12. Thalassemia
What it measures: Blood disorder affecting hemoglobin

Why it matters: Certain types can affect heart function and oxygen delivery.
Clinical progression (increasing concern):

Normal: No blood disorder

Reversible Defect: Temporary blood flow issues

Fixed Defect: Permanent damage to heart muscle

#### Medical insight: Fixed defects suggest previous heart attacks or permanent damage.

### 13. Vessels Colored by Fluoroscopy
What it measures: Number of major coronary arteries visible during angiography

Why it matters: More visible vessels often indicate more severe disease.
Clinical progression (increasing concern):

Zero: Minimal artery disease

### Interpret Results
Very Low Risk (<5%): Maintain healthy lifestyle

Low Risk (5-10%): Routine follow-up recommended

Borderline (10-20%): Consider cardiovascular assessment

Intermediate (20-40%): Further evaluation recommended

High Risk (>40%): Medical evaluation advised

One: Mild disease in one artery

Two: Moderate disease in multiple arteries

Three: Severe multi-vessel disease

### Actual ↓ / Predicted → | No Disease (0) | Disease (1)

No Disease (0)         | 11 (TN)        | 11 (FP)

Disease (1)            | 1 (FN)         | 1 (TP)

Disclaimer: This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions you may have regarding medical conditions.

