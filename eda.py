# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(filepath):
    """Loads the dataset, drops missing critical values, and removes duplicates."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['total_seconds', '5km_seconds', 'age_category'])
    df = df.drop_duplicates()
    return df

def get_summary_stats(df):
    """Returns a statistical summary of the continuous numerical columns."""
    return df[['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds']].describe()

def plot_univariate(df, bins=30):
    """Generates a histogram of total finish times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='total_seconds', kde=True, color='skyblue', bins=bins, ax=ax)
    ax.set_title('Univariate: Distribution of Overall Finish Times')
    ax.set_xlabel('Total Finish Time (Seconds)')
    ax.set_ylabel('Runner Count')
    plt.tight_layout()
    return fig

def plot_bivariate(df):
    """Generates a boxplot comparing finish times by age and sex."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df['age_category'].unique())
    sns.boxplot(data=df, x='age_category', y='total_seconds', hue='sex', order=order, ax=ax)
    ax.set_title('Bivariate: Finish Times Across Age and Gender Categories')
    ax.set_xlabel('Age Category')
    ax.set_ylabel('Total Finish Time (Seconds)')
    plt.tight_layout()
    return fig

def plot_correlation(df):
    """Generates a correlation heatmap of the race splits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_matrix = df[['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", square=True, ax=ax)
    ax.set_title('Correlation Matrix of Race Splits')
    plt.tight_layout()
    return fig

def plot_multivariate(df):
    """Generates a regression plot of the 5km split vs total finish time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Sample data to prevent the browser from crashing rendering 30k dots
    sample_df = df.sample(min(1000, len(df)))
    sns.regplot(data=sample_df, x='5km_seconds', y='total_seconds',
                scatter_kws={'alpha':0.3, 'color':'gray'},
                line_kws={'color':'blue', 'label':'Average Pacing Trend'}, ax=ax)
    ax.set_title('Multivariate/Efficiency Analysis: 5km Split vs. Final Time')
    ax.set_xlabel('5km Split Time (Seconds)')
    ax.set_ylabel('Total Finish Time (Seconds)')
    ax.legend()
    plt.tight_layout()
    return fig