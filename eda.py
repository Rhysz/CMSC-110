import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def format_time(x, pos):
    """Formats axis labels from raw seconds to MM:SS"""
    mins = int(x // 60)
    secs = int(x % 60)
    return f"{mins}:{secs:02d}"


def clean_data(filepath):
    """Loads data, removes invalid entries, and calculates pacing strategy."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['total_seconds', '5km_seconds', 'age_category'])
    df = df.drop_duplicates()

    # Filter out physically impossible timing errors (5K time >= Total time)
    df = df[df['total_seconds'] > df['5km_seconds']]

    # Calculate split strategy (Positive vs Negative)
    df['second_half_seconds'] = df['total_seconds'] - df['5km_seconds']
    df['split_strategy'] = df.apply(
        lambda x: 'Negative Split (Faster 2nd Half)' if x['second_half_seconds'] <= x['5km_seconds']
        else 'Positive Split (Slower 2nd Half)', axis=1
    )
    return df


def get_summary_stats(df):
    return df[['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds']].describe()


def plot_univariate(df, bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='total_seconds', kde=True, color='skyblue', bins=bins, ax=ax)
    ax.set_title('Univariate Analysis: Distribution of Overall Finish Times')
    ax.set_xlabel('Total Finish Time (MM:SS)')
    ax.set_ylabel('Runner Count')
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig


def plot_bivariate(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df['age_category'].unique())
    sns.boxplot(data=df, x='age_category', y='total_seconds', hue='sex', order=order, ax=ax)
    ax.set_title('Bivariate Analysis: Finish Times Across Age and Gender')
    ax.set_xlabel('Age Category')
    ax.set_ylabel('Total Finish Time (MM:SS)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    cols_to_correlate = ['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds']
    corr_matrix = df[cols_to_correlate].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", square=True, ax=ax)
    ax.set_title('Correlation Matrix of Race Splits')
    plt.tight_layout()
    return fig


def plot_efficiency(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df.sample(min(1000, len(df)))

    sns.scatterplot(data=sample_df, x='5km_seconds', y='total_seconds', hue='split_strategy',
                    palette={'Negative Split (Faster 2nd Half)': 'green', 'Positive Split (Slower 2nd Half)': 'red'},
                    alpha=0.6, ax=ax)

    ax.set_title('Multivariate Analysis: Pacing Efficiency (5km Split vs. Final Time)')
    ax.set_xlabel('5km Split Time (MM:SS)')
    ax.set_ylabel('Total Finish Time (MM:SS)')
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig


def plot_gender_gap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df['age_category'].unique())

    sns.barplot(data=df, x='age_category', y='total_seconds', hue='sex',
                estimator='median', errorbar=None, order=order, palette='muted', ax=ax)

    ax.set_title('Comparative Analysis: Median Finish Time (Male vs Female by Age)')
    ax.set_xlabel('Age Category')
    ax.set_ylabel('Median Finish Time (MM:SS)')
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig