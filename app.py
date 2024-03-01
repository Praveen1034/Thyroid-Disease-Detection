# Make Web Page
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# list
sex = ['Female', 'Male']
pregnant = ['True', 'False']

try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model file 'model.pkl' not found. Please ensure the model file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    # log the exception for further investigation
    raise

# web site view
st.title('Thyroid Disease Detection')
# Use columns to place selectors side by side
col1, col2 ,col3 = st.columns(3)

# 'Select The Gender Of Patient' dropdown
with col1:
    gender_patient = st.selectbox('Select The Gender Of Patient', sorted(sex))

# 'Is the patient pregnant?' dropdown
with col2:
    # Disable and set to 'False' if gender is 'male'
    if gender_patient == 'Male':
        female_pregnant = 'False'
        st.selectbox('Is the patient pregnant?', [female_pregnant], index=0, key='pregnant_selector')
    else:
        female_pregnant = st.selectbox('Is the patient pregnant?', sorted(pregnant))
        
with col3:
    Age = st.number_input('Age Of the patient', min_value=1, max_value=100, value=30)


TT4 = st.slider('TT4 Level In The Blood', min_value=2.0, max_value=209.0, value=106.0)


T3 = st.slider('T3 Level In The Blood', min_value=0.05, max_value=4.3, value=2.4)

T4U = st.slider('T4U Level In The Blood', min_value=0.19, max_value=1.54, value=1.06)

FTI = st.slider('FTI Level In The Blood', min_value=2.0, max_value=203.0, value=85.0)

TSH = st.slider('TSH Level In The Blood', min_value=-18.824586, max_value=74.0, value=1.60)



if st.button('Predict Probability'):
    input_data = pd.DataFrame({
        'age' : [Age],
        'sex': [gender_patient],
        'TT4': [TT4],
        'T3': [T3],
        'T4U': [T4U],
        'FTI': [FTI],
        'TSH': [TSH],
        'pregnant': [female_pregnant]
    })

    # Apply label encoding to 'sex' and 'pregnant' columns
    label_encoder = LabelEncoder()
    input_data['sex'] = label_encoder.fit_transform(input_data['sex'])
    input_data['pregnant'] = label_encoder.fit_transform(input_data['pregnant'])

    try:
        # Make sure to convert input_data to the same format as x_test used during training
        result_prob = model.predict(input_data)  # Assuming binary classification
        if result_prob == 0:
            st.markdown("<h2 style='color: red;'>The Patient Has Hyperthyroid Problem</h2>", unsafe_allow_html=True)
        elif result_prob == 1:
            st.markdown("<h2 style='color: red;'>The Patient Has Hypothyroid Problem</h2>", unsafe_allow_html=True)
        elif result_prob == 2:
            st.markdown("<h2 style='color: green;'>The Patient Has Negative</h2>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error predicting Probability: {e}")
        # Print the exception details
        print(f"Exception: {e}")
        # Raise the exception again for further investigation
        raise
    