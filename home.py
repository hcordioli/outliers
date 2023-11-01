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

bxp_url = "https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097"
zsc_url = "https://assess.com/z-score/#:~:text=Z%2Dscores%20generally%20range%20from,of%20the%20normal%20distribution%20curve."
mad_url = "https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/other-measures-of-spread/a/mean-absolute-deviation-mad-review"

st.markdown(
    """
    The Outlier Detector tool helps you find outliers in your dataset
    through the following steps:

        1. 📂 Load your dataset
        2. 👈 Chose a numerical column that you want to analyze
        3. 🪥 Chose the method you want to apply to remove possible \
inexistent values
        4. 📊 See the findings and decide what to do with them
            This version shows:
                * Statistics of original data
                * Boxplot
                * Z Score
                * Median Absolute deviation (MAD) Algorithm

    Below are some links with the theory behind.    
    I hope you enjoy.
"""
)
st.markdown("   * [Boxplot](%s)" % bxp_url)
st.markdown("   * [Z Score](%s)" % zsc_url)
st.markdown("   * [MAD](%s)" % mad_url)