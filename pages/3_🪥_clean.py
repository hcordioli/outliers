"""_Clean feature column
Allows the user to identify missing data and repair them by one of three options:
    1. Delete
    2. Replace by Mean
    3. Keep
"""
import streamlit as st


def apply_change():
    """Callback function to handle missing data procedure
    """
    if st.session_state['cleanup_method'] == "Delete":
        st.session_state['selected_serie'] = st.session_state['selected_serie'][st.session_state['selected_serie'].notnull()]
    elif st.session_state['cleanup_method'] == "Replace with mean":
        st.session_state['selected_serie'] = st.session_state['selected_serie'].fillna(st.session_state['selected_serie'].mean())
    else:
        st.session_state['selected_serie'] = st.session_state['selected_serie']

if 'selected_feature' not in st.session_state:
    st.warning("You must have picked a feature. Click on Choose Tab to do that first.")
    st.stop()  # App won't run anything after this line

nan_rows_df = st.session_state['selected_serie'][st.session_state['selected_serie'].isna()]

st.write(f"There are {len(nan_rows_df)} rows with no values in {st.session_state['selected_feature']} feature")
cleanup_method = st.selectbox(
    'How would you like to address ?',
    ["Keep", "Delete", "Replace with mean"], on_change=apply_change, key="cleanup_method")
