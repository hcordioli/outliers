"""Choose a feature/dataset column page
Allows the user to choose the feature to be analyzed for
anomaly/outlier detection
"""
import streamlit as st
import numpy as np

if 'dataframe' not in st.session_state:
    st.warning("You must have uploaded a file. Click on Load Tab to do that first.")
    st.stop()  # App won't run anything after this line
#elif 'selected_feature' not in st.session_state:
#    st.warning("You must have picked a feature. Click on Choose Tab to do that first.")
#    st.stop()  # App won't run anything after this line


num_df = st.session_state['dataframe'].select_dtypes(include=np.number)
features = num_df.columns
if 'selected_feature' not in st.session_state:
    st.session_state['selected_feature'] = None
if 'selected_serie' not in st.session_state:
    st.session_state['selected_serie'] = None

selected_feature = st.selectbox(
    'What feature you would like to analyze?',
    features)

st.write('You selected:', selected_feature)
st.session_state['selected_feature'] = selected_feature
st.session_state['selected_serie'] = st.session_state['dataframe'][st.session_state['selected_feature']]
if 'outliers_df' in st.session_state:
    del st.session_state['outliers_df']
