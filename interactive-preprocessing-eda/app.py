import streamlit as st
import pandas as pd
from utils.preprocessing import handle_missing_values, remove_outliers, scale_features
from utils.eda import generate_eda_report

# App Title
st.title("🔍 Interactive EDA & Preprocessing Web App")

# Upload dataset
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('📌 Raw Dataset')
    st.write(df.head())

    # EDA
    if st.button('Generate EDA Report'):
        eda_report = generate_eda_report(df)
        st.subheader('📊 EDA Summary')
        st.write(eda_report)

    # Preprocessing Options
    st.subheader('🔧 Preprocessing Options')
    missing_options = st.checkbox('Handle Missing Values')
    encode_option = st.checkbox('Encode Categorical Data')
    outlier_option = st.checkbox('Remove Outliers')
    scale_option = st.checkbox('Scale Features')

    # Initialize processed_df to avoid undefined errors
    processed_df = df.copy()
    st.subheader('🛠 Handle Missing Values')
    drop_na = st.checkbox('Drop Rows With Missing Values')
    numeric_strategy = st.radio(
        'Select strategy for numeric columns:',
        ('mean', 'median', 'mode')
    )
    if st.button('Apply Preprocessing'):
        if missing_options:
            processed_df = handle_missing_values(df, numeric_strategy, drop_na)
        # if encode_option:
        # processed_df = encode_categorical(df)
        if outlier_option:
            processed_df = remove_outliers(df)
        if scale_option:
            processed_df = scale_features(df)

    # Download preprocessed Data
    csv = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Preprocessed  CSV', csv, 'processed_data.csv', 'text/csv')
