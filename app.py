import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Page configuration
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    train_df = pd.read_csv('Training.csv').drop(columns='Unnamed: 133')
    test_df = pd.read_csv('Testing.csv')
    return train_df, test_df

@st.cache_resource
def load_model():
    model_data = joblib.load('disease_prediction_xgboost.pkl')
    return model_data

train_df, test_df = load_data()
model_data = load_model()
model = model_data['model']
encoder = model_data['label_encoder']
features = model_data['feature_names']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Option", ["EDA", "Predictions"])

if page == "EDA":
    # EDA Section
    st.title("Exploratory Data Analysis")
    
    tab1, tab2 = st.tabs(["Training Data", "Testing Data"])
    
    with tab1:
        st.subheader("Training Data Overview")
        st.write(f"Shape: {train_df.shape}")
        st.write("First 5 rows:")
        st.dataframe(train_df.head())
        
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        train_df['prognosis'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        st.subheader("Missing Values")
        st.write(train_df.isnull().sum())
        
        st.subheader("Symptom Frequency")
        symptoms = train_df.columns[:-1]
        symptom_counts = train_df[symptoms].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 20))
        symptom_counts.plot(kind='barh', ax=ax)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Testing Data Overview")
        st.write(f"Shape: {test_df.shape}")
        st.write("First 5 rows:")
        st.dataframe(test_df.head())
        
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        test_df['prognosis'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        st.subheader("Missing Values")
        st.write(test_df.isnull().sum())

else:
    # Prediction Section
    st.title("Disease Prediction System")
    
    st.write("""
    ### Enter Symptoms
    Select 'Yes' for symptoms you're experiencing and 'No' for those you're not.
    """)
    
    # Create a dictionary to store symptom values
    symptoms_dict = {feature: 0 for feature in features}
    
    # Organize symptoms into columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Split symptoms into 3 columns
    symptoms_per_col = len(features) // 3
    for i, symptom in enumerate(features):
        if i < symptoms_per_col:
            with col1:
                symptoms_dict[symptom] = st.selectbox(
                    symptom.replace('_', ' ').title(),
                    ['No', 'Yes'],
                    key=symptom
                )
        elif i < 2*symptoms_per_col:
            with col2:
                symptoms_dict[symptom] = st.selectbox(
                    symptom.replace('_', ' ').title(),
                    ['No', 'Yes'],
                    key=symptom
                )
        else:
            with col3:
                symptoms_dict[symptom] = st.selectbox(
                    symptom.replace('_', ' ').title(),
                    ['No', 'Yes'],
                    key=symptom
                )
    
    # Convert 'Yes'/'No' to 1/0
    symptoms_input = {k: 1 if v == 'Yes' else 0 for k, v in symptoms_dict.items()}
    
    # Prediction button
    if st.button("Predict Disease"):
        # Convert to array in correct order
        symptoms_array = np.array([symptoms_input[feature] for feature in features]).reshape(1, -1)
        
        # Make prediction
        pred_encoded = model.predict(symptoms_array)
        disease = encoder.inverse_transform(pred_encoded)[0]
        probabilities = model.predict_proba(symptoms_array)[0]
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_diseases = encoder.inverse_transform(top3_indices)
        top3_probs = probabilities[top3_indices]
        
        # Display results
        st.success(f"### Primary Prediction: {disease}")
        
        st.subheader("Top 3 Predictions")
        for disease, prob in zip(top3_diseases, top3_probs):
            st.write(f"- {disease} ({(prob*100):.1f}%)")
        
        # Show all probabilities in an expandable section
        with st.expander("View all disease probabilities"):
            prob_df = pd.DataFrame({
                'Disease': encoder.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
        
        # Show important symptoms for the prediction
        st.subheader("Key Symptoms Contributing to Prediction")
        importance = model.feature_importances_
        important_features = pd.DataFrame({
            'Symptom': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Symptom', data=important_features, ax=ax)
        ax.set_yticklabels([s.replace('_', ' ').title() for s in important_features['Symptom']])
        st.pyplot(fig)

# Add some styling
st.markdown("""
<style>
    .stSelectbox label, .stButton button {
        font-size: 16px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        margin: 10px 0;
    }
    .stSuccess {
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)