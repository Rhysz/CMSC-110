'''
BAGORIO, CAPULE, DIZON, RINON
2026 - 05
Description: Webapp using streamlit library
'''

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import eda
import os

# 1. Title and Introduction
st.set_page_config(page_title="Madrid 10K Analyzer", layout="wide")
st.title("🏃‍♂️ San Silvestre Vallecana 2019 Performance Analyzer")

st.write("### Purpose and Importance of the Application")
st.write(
    "This application analyzes finisher data from the Madrid New Year's Eve 10K race. It is designed to help runners, coaches, and sports analysts investigate pacing strategies, understand demographic performance trends, and predict placements across varying age brackets and gender demographics."
)


# 2. Data Loading
@st.cache_data
def load_dataset():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_directory, 'madrid_10k_20191231.csv')
    return eda.clean_data(csv_path)


try:
    df = load_dataset()
except Exception as e:
    st.error(f"⚠️ Dataset mapping error: {e}")
    st.stop()

# 3. Sidebar Filtering
st.sidebar.header("🎯 Live Filter Controls")
st.sidebar.markdown("Use these adjustments to segment the race field:")

gender_list = sorted(list(df['sex'].unique()))
selected_genders = st.sidebar.multiselect("Select Gender Group:", gender_list, default=gender_list)

age_categories = sorted([cat for cat in df['age_category'].dropna().unique()])
selected_ages = st.sidebar.multiselect("Select Age Categories:", age_categories, default=age_categories)

min_min = float(np.floor(df['total_minutes'].min()))
max_min = float(np.ceil(df['total_minutes'].max()))
selected_time_range = st.sidebar.slider(
    "Set Completion Time Window (Minutes):",
    min_value=min_min,
    max_value=max_min,
    value=(min_min, max_min),
    step=0.5
)

filtered_df = df[
    (df['sex'].isin(selected_genders)) &
    (df['age_category'].isin(selected_ages)) &
    (df['total_minutes'] >= selected_time_range[0]) &
    (df['total_minutes'] <= selected_time_range[1])
    ]

if filtered_df.empty:
    st.warning(
        "⚠️ No data items match your exact selected filter targets. Adjust the sidebar sliders or choose more groups!")
    st.stop()

# 4. Key Performance Metrics
st.subheader("📊 Segment Overview & Key Metrics")
col1, col2, col3, col4 = st.columns(4)

avg_seconds = filtered_df['total_seconds'].mean()
fastest_seconds = filtered_df['total_seconds'].min()

col1.metric("Competitors in View", f"{filtered_df.shape[0]:,}")
col2.metric("Mean Completion Pace", f"{int(avg_seconds // 60)}:{int(avg_seconds % 60):02d}")
col3.metric("Mean Halfway Split (5K)", f"{filtered_df['split_5k_minutes'].mean():.1f} mins")
col4.metric("Fastest Clock Time", f"{int(fastest_seconds // 60)}:{int(fastest_seconds % 60):02d}")

st.divider()

# 5. Prediction Form
st.subheader("⏱️ Placement Prediction Calculator")
st.write("Enter your target finish time to see your predicted percentile rank based on the filtered cohort.")

form_col1, form_col2 = st.columns(2)
with form_col1:
    target_mins = st.number_input("Target Minutes", min_value=20, max_value=120, value=55)
with form_col2:
    target_secs = st.number_input("Target Seconds", min_value=0, max_value=59, value=0)

total_target_seconds = (target_mins * 60) + target_secs

if len(filtered_df) > 0:
    percentile = stats.percentileofscore(filtered_df['total_seconds'], total_target_seconds, kind='strict')
    beat_percentage = 100 - percentile

    if percentile <= 25:
        st.success(
            f"**Top Tier!** You would be faster than {beat_percentage:.1f}% of the runners in this category. :orange[(**Top {percentile:.1f}%**)]")
    elif percentile <= 50:
        st.info(
            f"**Above Average!** You would be faster than {beat_percentage:.1f}% of the runners in this category. :green[(**Top {percentile:.1f}%**)]")
    else:
        st.warning(f"**Keep Pushing!** You would be in the :red[**bottom {100 - percentile:.1f}%**] of this category.")

st.divider()

# 6. Interactive Visualizations
st.subheader("📈 Exploratory Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Finish Time Distribution",
    "Demographics",
    "Pacing Efficiency",
    "Split Correlation"
])

with tab1:
    st.write("**Univariate Analysis:** Adjust the slider to change the granularity of the histogram.")
    bins = st.slider("Number of Histogram Bins", min_value=10, max_value=100, value=30)
    fig_uni = eda.plot_univariate(filtered_df, bins=bins)
    st.pyplot(fig_uni)

with tab2:
    st.write("**Bivariate Analysis:** Comparing finish times across age and gender.")
    if len(filtered_df) > 0:
        fig_bi = eda.plot_bivariate(filtered_df)
        st.pyplot(fig_bi)

with tab3:
    st.write("**Multivariate Analysis:** Efficiency and pacing strategy. Green points indicate a faster second half.")
    if len(filtered_df) > 10:
        fig_multi = eda.plot_efficiency(filtered_df)
        st.pyplot(fig_multi)

with tab4:
    st.write(
        "**Interactive Correlation Analysis:** Discover how strongly early race splits predict final finish times.")

    col5, col6 = st.columns([1.2, 1])
    with col5:
        # Mapping clean text labels to the raw dataframe columns
        metric_mapping = {
            '2.5km Split Time': '2.5km_seconds',
            '5km Split Time': '5km_seconds',
            '7.5km Split Time': '7.5km_seconds'
        }

        chosen_label = st.selectbox(
            "Select intermediate benchmark to correlate with total finish time:",
            list(metric_mapping.keys())
        )
        chosen_metric = metric_mapping[chosen_label]

        if len(filtered_df) > 5:
            fig_corr = eda.plot_correlation(filtered_df, chosen_metric, chosen_label)
            st.pyplot(fig_corr)

    with col6:
        st.markdown(f"""
        <br><br>
        **Interactive Correlation Insight:**
        * You are currently inspecting the interactive link between **{chosen_label}** and final finish time.
        * A coefficient near **1.0000** indicates that performance at that specific checkpoint strongly anchors the eventual placement. 
        * Changing the dropdown allows you to witness mathematically how pacing correlation strengthens or stabilizes as athletes approach the final marker.
        """, unsafe_allow_html=True)