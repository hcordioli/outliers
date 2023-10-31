""" Present all the charts indicating outliers and
    nature of the features distributions
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import zscore
from pyod.models.mad import MAD

os = st.session_state['selected_serie']
if 'outliers_df' not in st.session_state:
    st.session_state['outliers_df'] = pd.DataFrame(os)

OUTPUT_FILENAME = "outliers.csv"


def save_file():
    """Save the Outliers Dataset in a file
    """
    st.session_state['outliers_df'].to_csv(OUTPUT_FILENAME, index=False)


st.set_page_config(layout="wide")


# Boxplot Outliers
def get_bxp_outliers(data, bxp_whis):
    """Calculate the outliers from a sample based on the whis parameter

    Args:
        data (Pandas Series): Tha sample
        bxp_whis (float): Whis parameter

    Returns:
        Pandas Series: The ouliers
    """
    first_quartile = 0.25
    third_quartile = 0.75
    quartile_1 = data.quantile(first_quartile)
    quartile_3 = data.quantile(third_quartile)
    iqr = quartile_3 - quartile_1
    lower_limit = quartile_1 - (iqr * bxp_whis)
    upper_limit = quartile_3 + (iqr * bxp_whis)
    is_lower = data < lower_limit
    is_higher = data > upper_limit
    result = pd.Series(
        data = list(map(lambda i:  i < lower_limit or i > upper_limit, data)),
        name = st.session_state['selected_feature']
    )

    return result
#    return data[is_lower | is_higher].sort_values()

cont_org = st.container()
c_no_5, c_no_h, c_no_p, c_no_s, c_no_a = st.columns(
    5,
    gap="small"
)
cont_bxp = st.container()
c_bp_5, c_bp_h, c_bp_p, c_bp_s, c_bp_af, c_bp_as = st.columns([
    1.8, 1.8, 1.8, 1.8, 1.8, 1
])
cont_zsc= st.container()
c_zs_5, c_zs_h, c_zs_p, c_zs_s, c_zs_a = st.columns(
    5,
    gap="small"
)
cont_mad = st.container()
c_ma_5, c_ma_h, c_ma_p, c_ma_s, c_ma_a = st.columns(
    5,
    gap="small"
)


def plot_bxp(input_df, container, input_whis, num_outliers):
    """Plot the Box Plot widget
    Args:
        df (Pandas Series): Data to be plotted
        container (Streamlit Container): Plotting location
        input_whis (Float): Parameter to calculate Inter Quartile Range
        n_outliers (int): Number of Outliers after calculation
    """
    msg = " outlier " if num_outliers == 0 else " outliers"
    container.write(f'{num_outliers}{msg}')
    fig_bxp, _ = plt.subplots(figsize=(3, 7))
    sns.boxplot(
        data=input_df,
        # Apparently a bug in Seaborn forces such transformation
        # Must be a float number, not rounded
        y=input_df.values/1.0001, 
        whis=input_whis,
        orient='y',
        width=0.3
    )
    container.pyplot(fig_bxp)


def plot_hist(input_df, container):
    """Plot Histogram widget

    Args:
        input_df (Pandas Series): Data to be plotted
        container (Streamlit Container): Plotting Location
    """
    num_bins = int(np.sqrt(len(input_df)))
    fig_hist = px.histogram(
        input_df,
        x=st.session_state['selected_feature'],
        nbins=num_bins
    )
    container.plotly_chart(fig_hist, use_container_width=True)


def plot_pdf(input_df, container):
    """Plot Probability Distribution Function

    Args:
        input_df (Pandas Series): Data to be plotted
        container (Streamlit Container): Plotting Location
    """
    feat_mu = input_df.mean()
    feat_std = input_df.std()
    feat_pdf = norm.pdf(input_df.values, feat_mu, feat_std)
    fig_pdf = px.scatter(
        x=input_df,
        y=feat_pdf,
        labels={'x': st.session_state['selected_feature'], 'y': 'pdf'})
    fig_pdf.update_traces(marker_size=1)
    container.plotly_chart(fig_pdf, use_container_width=True)


def plot_scatterplot(input_df, container):
    """Plot Scatterplot widget
        How the choosed feature gets distributed in terms of occurrence
    Args:
        input_df (Pandas Serie): Data to be plotted
        container (Streamlit Container): Plotting Location
    """
    intervals = range(len(input_df))
    fig_scatter = px.scatter(
        x=intervals,
        y=input_df,
        labels={'x': 'Intervals', 'y': st.session_state['selected_feature']})
    fig_scatter.update_traces(opacity=.4, marker_size=2)
    container.plotly_chart(fig_scatter, use_container_width=True)


# Original Data
with st.container():
    cont_org.write("Statistics of the Original Data")
    # The 5 numbers
    c_no_5.table(os.describe().apply("{0:.2f}".format))

    # Histogram
    plot_hist(os, c_no_h)

    # PDF
    plot_pdf(os, c_no_p)

    # Scatterplot
    plot_scatterplot(os, c_no_s)

    # We'll use Adjust column to inform about the selected feature
    result = shapiro(os)
    c_no_a.write(f'Feature Statistic: {result.statistic:,.2f}')
    c_no_a.write(f'Feature P-Value: {result.pvalue:,.2f}')
    P_VALUE_LIMIT = 0.05
    if result.pvalue < P_VALUE_LIMIT:
        c_no_a.write("It does not come from a normal distribution")
    else:
        c_no_a.write("Likely coming from a normal distribution")

# Box PLot
with st.container():
    cont_bxp.write("Boxplot Outliers")
    DEFAULT_WHIS = 1.5

    def calculate_bxp():
        """
            Display Boxplot chart
        """
        # Define the outliers by changing whis parameter

        new_whis = DEFAULT_WHIS if ("bxp_whis" not in st.session_state) else st.session_state['bxp_whis']
        bxp_outliers = get_bxp_outliers(os, new_whis)
        st.session_state['outliers_df']['bxp_outlier'] = bxp_outliers
        
    whis = DEFAULT_WHIS if ("bxp_whis" not in st.session_state) else st.session_state['bxp_whis']

    if 'bxp_whis' not in st.session_state:
        calculate_bxp()

    c_bp_df = st.session_state['outliers_df'].loc[st.session_state['outliers_df']['bxp_outlier'] == False][st.session_state['selected_feature']]
    n_outliers = len(st.session_state['outliers_df'].loc[st.session_state['outliers_df']['bxp_outlier'] == True])

    #plot_bxp(os, c_bp_af, whis, n_outliers)
    plot_bxp(st.session_state['dataframe'][st.session_state['selected_feature']], c_bp_af, whis, n_outliers)
    # The 5 numbers
    c_bp_5.table(c_bp_df.describe().apply("{0:.2f}".format))

    # Histogram
    plot_hist(c_bp_df, c_bp_h)

    # PDF
    plot_pdf(c_bp_df, c_bp_p)

    # Scatterplot
    plot_scatterplot(c_bp_df, c_bp_s)

    # The Whis Selector Slider

    whis_selector = c_bp_as.slider(
        'Select whis:',
        min_value=0.1,
        max_value=4.0,
        value=DEFAULT_WHIS,
        step=0.1,
        key="bxp_whis",
        on_change=calculate_bxp
    )

# Z Score
with st.container():
    cont_zsc.write("Z Score Outliers")
    # Adjust Z_Score threshold
    DEFAULT_Z_SCORE = 4.0

    def calculate_zs():
        """Calculate Z_Score
        """

        z_score_t = DEFAULT_Z_SCORE if ("zsc_threshold" not in st.session_state) else st.session_state['zsc_threshold']
        feature_zscore = zscore(st.session_state['selected_serie'])
        is_over_zt = np.abs(feature_zscore) > z_score_t
        z_outliers = st.session_state['selected_serie'][is_over_zt]
        msg = " outlier " if len(z_outliers) == 0 else " outliers"
        c_zs_a.write(f'{len(z_outliers)}{msg}')

        # Update outliers file
        st.session_state['outliers_df']['zsc_outlier'] = is_over_zt


    calculate_zs()

    # The 5 numbers
    non_outliers = st.session_state['outliers_df'].loc[st.session_state['outliers_df']['zsc_outlier'] == False][st.session_state['selected_feature']]
    c_zs_5.table(non_outliers.describe().apply("{0:.2f}".format))

    # Histogram
    plot_hist(non_outliers, c_zs_h)

    # PDF
    plot_pdf(non_outliers, c_zs_p)

    # Scatterplot
    plot_scatterplot(non_outliers, c_zs_s)

    zs_selector = c_zs_a.slider(
        'Select Z_Score Threshold:',
        min_value=0.1,
        max_value=4.0,
        value=DEFAULT_Z_SCORE,
        step=0.1,
        key="zsc_threshold"
        # on_change = calculate_zs
    )

# MAD Score
with st.container():
    cont_mad.write("MAD Outliers")
    # Adjust MAD threshold
    DEFAULT_MAD_THRESHOLD = 4.5

    def calculate_mad():
        """Use the median_abs_deviation to classify the selected feature as outliers
        """
        mad_threshold_t = DEFAULT_MAD_THRESHOLD if ("mad_threshold" not in st.session_state) else st.session_state['mad_threshold']
        mad = MAD(threshold=mad_threshold_t)

        # Reshape for Pyod
        feature_reshaped = st.session_state['selected_serie'].values.reshape(-1, 1)
        mad_classification = mad.fit(feature_reshaped)
        mad_df = pd.DataFrame({
            st.session_state['selected_feature']: st.session_state['selected_serie'],
            "mad_outlier": mad_classification.labels_
        })
        mad_outliers = mad_df.loc[mad_df['mad_outlier'] == 1]
        msg = " outlier " if len(mad_outliers) == 0 else " outliers"
        c_ma_a.write(f'{len(mad_outliers)}{msg}')

        # Update outliers file
        st.session_state['outliers_df']['mad_outlier'] = mad_classification.labels_
        st.session_state['outliers_df']['mad_outlier'] = st.session_state['outliers_df']['mad_outlier'].astype('bool')

    calculate_mad()

    non_outliers = st.session_state['outliers_df'].loc[st.session_state['outliers_df']['mad_outlier'] == False][st.session_state['selected_feature']]

    # The 5 numbers
    c_ma_5.table(non_outliers.describe().apply("{0:.2f}".format))

    # Histogram
    plot_hist(non_outliers, c_ma_h)

    # PDF
    plot_pdf(non_outliers, c_ma_p)

    # Scatterplot
    plot_scatterplot(non_outliers, c_ma_s)

    mad_selector = c_ma_a.slider(
        'Select MAD Threshold:',
        min_value=0.1,
        max_value=5.0,
        value=DEFAULT_MAD_THRESHOLD,
        step=0.1,
        key="mad_threshold",
        on_change=calculate_mad
    )

# Save File
with st.container():
    st.button("Save File", type="primary", on_click=save_file)
