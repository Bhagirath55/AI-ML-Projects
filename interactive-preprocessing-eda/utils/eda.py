import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def generate_eda_report(df):
    report = {}

    # Shape
    report['Shape'] = df.shape

    # Missing Values
    report['Missing Values'] = df.isnull().sum().to_dict()

    # Data Types
    report['Data Types'] = df.dtypes.to_dict()

    # Correlation heatmap
    st.subheader('📊 Correlation Heatmap')
    fig,ax  = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(),annot=True,cmap='coolwarm',ax=ax)
    st.pyplot(fig)

    return report