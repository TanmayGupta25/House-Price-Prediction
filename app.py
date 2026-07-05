
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

st.set_page_config(page_title='House Price Prediction', page_icon='Ἶ0', layout='wide')

def load_metadata():
    with open('models/metadata.json', 'r') as f:
        return json.load(f)

metadata = load_metadata()

st.sidebar.title('Navigation')
page = st.sidebar.radio('Pages:', ['Ἶ0 Home', 'ὄA About Dataset', 'ᾑ6 About Model', 'Ὄ8 Prediction', '὆8‍ὄB Developer'])

if page == 'Ἶ0 Home':
    st.title('House Price Prediction System')
    st.subheader('Summer School 2026')
    st.markdown('**Developed by:** Tanmay Gupta')
    st.write('This system uses Machine Learning to predict house prices.')
    st.info(f'**Problem Type:** {metadata.get("Problem Type", "Regression")}')
    st.info(f'**Best Model:** {metadata.get("Best Model", "N/A")}')

elif page == 'ὄA About Dataset':
    st.title('About Dataset')
    st.write(f'**Dataset Name:** {metadata.get("Dataset Name", "Bengaluru House Data")}')
    st.write(f'**Target Column:** {metadata.get("Target Column", "price")}')

elif page == 'ᾑ6 About Model':
    st.title('About Model')
    st.write(f'**Best Model:** {metadata.get("Best Model", "N/A")}')
    st.metric('R2 Score', round(metadata.get('R2 Score', 0), 4))

elif page == 'Ὄ8 Prediction':
    st.title('House Price Prediction')
    model = joblib.load('models/best_model.pkl')
    features = joblib.load('models/feature_columns.pkl')
    input_data = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(features):
        target_col = col1 if i % 2 == 0 else col2
        if col in ['bath', 'balcony', 'total_sqft']:
            input_data[col] = target_col.number_input(f'{col}', value=0.0)
        else:
            input_data[col] = target_col.text_input(f'{col}', 'Unknown')
    if st.button('Predict'):
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        st.metric('Ἶ0 Estimated House Price', f'{round(prediction, 2)} Lakhs')

elif page == '὆8‍ὄB Developer':
    st.title('Developer Information')
    st.markdown('**Developer:** Tanmay Gupta')
    st.markdown('**Programme:** Summer School 2026')
