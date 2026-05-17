import streamlit as st
import pandas as pd
import scipy.stats as stats
import eda

# 1. Title and Introduction
st.set_page_config(page_title="Madrid 10K Analyzer", layout="wide")
st.title("🏃‍♂️ Madrid 10K Race Performance Analyzer")

st.write("### Brief Introduction")
st.write(
    "This application analyzes finisher data from the Madrid New Year's Eve 10K race. It is designed to help runners analyze historical pacing strategies, understand demographic performance trends, and predict their placement for future races based on real-world data.")


# 2. Data Loading
@st.cache_data
def load_dataset():
    return eda.clean_data("madrid_10k_20191231.csv")


df = load_dataset()

# 3. Sidebar Filtering
st.sidebar.header("Filter Race Data")

genders = ["All"] + list(df['sex'].unique())
age_groups = ["All"] + list(sorted(df['age_category'].unique()))

selected_gender = st.sidebar.selectbox("Select Gender", genders)
selected_age = st.sidebar.selectbox("Select Age Category", age_groups)

filtered_df = df.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df['sex'] == selected_gender]
if selected_age != "All":
    filtered_df = filtered_df[filtered_df['age_category'] == selected_age]

st.sidebar.markdown("---")
st.sidebar.write(f"**Runners matching criteria:** {len(filtered_df):,}")

# 4. Key Performance Metrics
st.subheader("📊 Key Performance Metrics")
col1, col2, col3 = st.columns(3)

if len(filtered_df) > 0:
    avg_seconds = filtered_df['total_seconds'].mean()
    fastest_seconds = filtered_df['total_seconds'].min()

    col1.metric("Total Runners Analyzed", f"{len(filtered_df):,}")
    col2.metric("Average Finish Time", f"{int(avg_seconds // 60)}:{int(avg_seconds % 60):02d}")
    col3.metric("Fastest Finish Time", f"{int(fastest_seconds // 60)}:{int(fastest_seconds % 60):02d}")
else:
    st.warning("No data matches the selected filters.")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Finish Time Distribution",
    "Demographics",
    "Split Correlation",
    "Pacing Efficiency",
    "Gender Pace Gap"
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
    st.write("**Correlation Analysis:** How strongly early race splits correlate with the final finish time.")
    if len(filtered_df) > 5:
        fig_corr = eda.plot_correlation(filtered_df)
        st.pyplot(fig_corr)

with tab4:
    st.write("**Multivariate Analysis:** Efficiency and pacing strategy. Green points indicate a faster second half.")
    if len(filtered_df) > 10:
        fig_multi = eda.plot_efficiency(filtered_df)
        st.pyplot(fig_multi)

with tab5:
    st.write("**Comparative Analysis:** Median finish times by gender across age brackets.")
    if len(filtered_df) > 0:
        fig_gap = eda.plot_gender_gap(filtered_df)
        st.pyplot(fig_gap)