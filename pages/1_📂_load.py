"""Load page
Allows the user to choose a file to be analyzed
"""

import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import sys
import numpy as np

sys.tracebacklimit = 0

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = None


def uploader_callback():

    """Callback function to handle file load
    """
    if st.session_state['file_uploader'] is not None:
        try:
            st.session_state['dataframe'] = pd.read_csv(
                st.session_state['file_uploader'],
                sep=st.session_state['csv_separator'])
        except pd.errors.ParserError as parser_error:
            st.write("An error occurred while parsing the CSV file")
            raise parser_error from None
        except pd.errors.EmptyDataError as empty_file_error:
            st.write("CSV file is empty")
            raise empty_file_error from None
        

sep_picker = st.selectbox(
    'What is CSV File separator?',
    (',', ';', '\t'),
    key="csv_separator"
)

df = st.file_uploader(
    label="File uploader", on_change=uploader_callback, key="file_uploader"
)

if st.session_state['dataframe'] is not None:
    num_df = st.session_state['dataframe']._get_numeric_data()        
    st.write(num_df.head())