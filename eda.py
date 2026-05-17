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
    """Loads data, removes invalid entries, calculates pacing, and formats labels."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['total_seconds', '5km_seconds', 'age_category', 'sex'])
    df = df.drop_duplicates()

    # Filter 1: Physically impossible timing errors (5K time >= Total time)
    df = df[df['total_seconds'] > df['5km_seconds']]

    # Calculate second half time for advanced filtering
    df['second_half_seconds'] = df['total_seconds'] - df['5km_seconds']

    # Filter 2: Remove impossible second-half times
    df = df[df['second_half_seconds'] > 750]

    # Filter 3: Remove extreme pace discrepancies
    df = df[(df['5km_seconds'] / df['second_half_seconds'] < 3) &
            (df['second_half_seconds'] / df['5km_seconds'] < 3)]

    # Calculate split strategy (Positive vs Negative)
    df['split_strategy'] = df.apply(
        lambda x: 'Negative Split (Faster 2nd Half)' if x['second_half_seconds'] <= x['5km_seconds']
        else 'Positive Split (Slower 2nd Half)', axis=1
    )

    # Convert seconds to minutes for the sidebar sliders
    df['total_minutes'] = df['total_seconds'] / 60
    df['split_5k_minutes'] = df['5km_seconds'] / 60

    # Standardize string formatting and rename genders
    df['sex'] = df['sex'].str.upper().str.strip().replace({'M': 'Male', 'F': 'Female'})

    return df


def plot_univariate(df, bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='total_seconds', kde=True, color='skyblue', bins=bins, ax=ax)

    mean_val = df['total_seconds'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {int(mean_val // 60)}:{int(mean_val % 60):02d}")

    ax.set_title('Univariate Analysis: Distribution of Overall Finish Times')
    ax.set_xlabel('10km Total Finish Time (MM:SS)')
    ax.set_ylabel('Runner Count')
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig


def plot_bivariate(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df['age_category'].unique())
    sns.boxplot(data=df, x='age_category', y='total_seconds', hue='sex', order=order, ax=ax, palette='Pastel1')

    ax.set_title('Bivariate Analysis: Finish Times Across Age and Gender')
    ax.set_xlabel('Age Category')
    ax.set_ylabel('10km Total Finish Time (MM:SS)')
    ax.legend(title='Gender')  # Cleaned legend title
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
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
    ax.set_ylabel('10km Total Finish Time (MM:SS)')
    ax.legend(title='Pacing Strategy')  # Cleaned legend title
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig


def plot_correlation(df, chosen_metric, chosen_label):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Temporarily rename columns just for the heatmap display
    display_df = df[[chosen_metric, 'total_seconds']].rename(
        columns={chosen_metric: chosen_label, 'total_seconds': '10km Total Time'}
    )

    corr_matrix = display_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", square=True, ax=ax)

    ax.set_title(f'Correlation Matrix: {chosen_label} vs 10km Total Time')
    plt.tight_layout()
    return fig