""" Outliers Detection Home page

The main page of the Outliers Detection tool
It provides a briefe explanation of the tool and the steps required to be used
"""

import streamlit as st

st.set_page_config(
    page_title="Outlier Detector",
    page_icon="🔍",
)

st.write("# Welcome to the Outlier Detector tool! 🔍")

st.markdown(
    """
    The Outlier Detector tool helps you find outliers in your dataset
    through the following steps:

        1. 📂 Load your dataset
        2. 👈 Chose a numerical column that you want to analyze
        3. 🪥 Chose the method you want to apply to remove possible \
inexistent values
        4. 📊 See the findings and decide what to do with them
            This version display:
                * Statistics of original data
                * Boxplot
                * Z Score
                * Median Absolute deviation (MAD) Algorithm

    I hope you enjoy.

"""
)
