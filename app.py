import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import eda
import os

# 1. Title and Introduction
st.set_page_config(page_title="Madrid 10K Analyzer", layout="wide")
st.title("🏃‍♂️ Madrid 10K Race Performance Analyzer")

# --- IMAGE 1: Hero Banner ---
st.image("runners.png", use_container_width=True, caption="San Silvestre Vallecana - The Final Race of the Year")

st.write("### Welcome to the ultimate finish line post-mortem")
st.write(
    "This application analyzes finisher data from the Madrid New Year's Eve 10K race. It is designed to help runners, coaches, and sports analysts investigate pacing strategies, understand demographic performance trends, and predict placements across varying age brackets and gender demographics."
)

with st.expander("📖 About the Race & Data Significance"):
    col_exp1, col_exp2 = st.columns([2, 1])

    with col_exp1:
        st.markdown("""
        * **The Route:** The race follows a linear path from the Santiago Bernabéu Stadium (Chamartín) to the Estadio de Vallecas.
        * **Elevation Profile:** The course is famous for its "fast but deceptive" layout. The first 8 kilometers are primarily downhill, encouraging high initial velocities. 
        * **The Sting in the Tail:** The final 2 kilometers feature a challenging uphill incline as runners enter the Vallecas district. This profile often results in the "Split Decay" seen in the Pacing Efficiency metrics.
        * **Competitive Field:** The event is divided into two distinct tiers:
            * *San Silvestre Popular:* A mass-participation event for over 40,000 amateur runners.
            * *San Silvestre Internacional:* An elite only reserved for athletes with proven sub-39 minute (men) or sub-45 minute (women) 10K times.
        * **Data Significance:** This dataset offers a unique look at human performance under pressure. Because the race takes place on the final day of the year, it captures peak performance data where runners are often attempting to set their final personal best of the season.
        """)
    with col_exp2:
        # --- IMAGE 2: Start Area / Event Context ---
        st.image("start.png", use_container_width=True, caption="Race Start at Santiago Bernabéu")


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

col1.metric("Total competitor", f"{filtered_df.shape[0]:,}",
            help="This represents the total number of athletes included in the current filtered dataset. This metric establishes the competitive scale and provides the necessary population volume to validate demographic performance trends.")
col2.metric("Mean completion time", f"{int(avg_seconds // 60)}:{int(avg_seconds % 60):02d}",
            help="The historical performance equilibrium for the field. This serves as the primary benchmark for a mid-pack finish, separating the top-tier competitive bracket from the recreational participant base.")
col3.metric("Mean Halfway Split (5K)", f"{filtered_df['split_5k_minutes'].mean():.1f} mins",
            help="The mean split recorded at the 5K interval. This metric tracks early-stage energy expenditure and serves as a critical indicator for pacing.")
col4.metric("Fastest Clock Time", f"{int(fastest_seconds // 60)}:{int(fastest_seconds % 60):02d}")

st.divider()

# 5. Prediction Form
st.subheader("⏱️ Placement Prediction Calculator")
st.write("Punch in your stats to see exactly where you land in the pack.")

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
st.subheader("📈 The Analytical Toolkit")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Finish Time Distribution",
    "Demographics",
    "Pacing Efficiency",
    "Split Correlation",
    "Route & Splits",
    "Raw Dataset"
])

with tab1:
    st.write("**Finish Time distribution:** See where the crowd clusters and how you compare to the 'average' runner.")
    bins = st.slider("Number of Histogram Bins", min_value=10, max_value=100, value=30)
    fig_uni = eda.plot_univariate(filtered_df, bins=bins)
    st.pyplot(fig_uni)

with tab2:
    st.write("**Demographics:** A deep dive into the age, gender, and regional composition of the race.")
    if len(filtered_df) > 0:
        fig_bi = eda.plot_bivariate(filtered_df)
        st.pyplot(fig_bi)

with tab3:
    st.write("**Pacing Efficiency:** Did you start too fast? Analyze Negative Splits vs. the Positive Splits.")
    if len(filtered_df) > 10:
        fig_multi = eda.plot_efficiency(filtered_df)
        st.pyplot(fig_multi)

with tab4:
    st.write("**Split Correlation:** See how your 5K split predicted your final 10K outcome.")

    col5, col6 = st.columns([1.2, 1])
    with col5:
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
        """, unsafe_allow_html=True)

with tab5:
    st.write("**Route & Splits:** The course profile mapped against median segment paces.")

    # --- IMAGE 3: Route Map ---
    st.image("route.png", use_container_width=True, caption="Race Route Map: Madrid 10K")

    if len(filtered_df) > 0:
        fig_route = eda.plot_route_splits(filtered_df)
        st.plotly_chart(fig_route, use_container_width=True)

with tab6:
    st.write("**Cleaned Dataset:** View the fully sanitized raw data (Missing splits and impossible times removed).")
    st.dataframe(filtered_df, use_container_width=True)