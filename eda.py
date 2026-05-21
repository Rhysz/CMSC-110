import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go


# Formats axis labels from raw seconds to MM:SS
def format_time(x, pos):
    mins = int(x // 60)
    secs = int(x % 60)
    return f"{mins}:{secs:02d}"

#Loads data, removes invalid entries, calculates pacing, and formats labels.
def clean_data(filepath):
    df = pd.read_csv(filepath)

    #Drop ANY row missing the 2.5km, 5km, 7.5km, or total time
    df = df.dropna(subset=['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds', 'age_category', 'sex'])
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

    # Convert all splits to minutes for the interactive charts and sliders
    df['split_2.5k_minutes'] = df['2.5km_seconds'] / 60
    df['split_5k_minutes'] = df['5km_seconds'] / 60
    df['split_7.5k_minutes'] = df['7.5km_seconds'] / 60
    df['total_minutes'] = df['total_seconds'] / 60

    # Standardize string formatting and rename genders
    df['sex'] = df['sex'].str.upper().str.strip().replace({'M': 'Male', 'F': 'Female'})

    return df

#For basic statistical metrics
def plot_univariate(df, bins=30):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='total_seconds', kde=True, color='skyblue', bins=bins, ax=ax)

    mean_val = df['total_seconds'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {int(mean_val//60)}:{int(mean_val%60):02d}")

    ax.set_title('Univariate Analysis: Distribution of Overall Finish Times')
    ax.set_xlabel('10km Total Finish Time (MM:SS)')
    ax.set_ylabel('Runner Count')
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig

#For filtered metrics (Age and Gender)
def plot_bivariate(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df['age_category'].unique())
    sns.boxplot(data=df, x='age_category', y='total_seconds', hue='sex', order=order, ax=ax, palette='Pastel1')

    ax.set_title('Bivariate Analysis: Finish Times Across Age and Gender')
    ax.set_xlabel('Age Category')
    ax.set_ylabel('10km Total Finish Time (MM:SS)')
    ax.legend(title='Gender')
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig

#For comparing 5km split time to 10km total time
def plot_efficiency(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df.sample(min(1000, len(df)))

    sns.scatterplot(data=sample_df, x='5km_seconds', y='total_seconds', hue='split_strategy',
                    palette={'Negative Split (Faster 2nd Half)': 'green', 'Positive Split (Slower 2nd Half)': 'red'},
                    alpha=0.6, ax=ax)

    ax.set_title('Multivariate Analysis: Pacing Efficiency (5km Split vs. Final Time)')
    ax.set_xlabel('5km Split Time (MM:SS)')
    ax.set_ylabel('10km Total Finish Time (MM:SS)')
    ax.legend(title='Pacing Strategy')
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    plt.tight_layout()
    return fig

#To see if faster splits equate to faster finish time
def plot_correlation(df, chosen_metric, chosen_label):
    fig, ax = plt.subplots(figsize=(6, 5))

    display_df = df[[chosen_metric, 'total_seconds']].rename(
        columns={chosen_metric: chosen_label, 'total_seconds': '10km Total Time'}
    )

    corr_matrix = display_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", square=True, ax=ax)

    ax.set_title(f'Correlation Matrix: {chosen_label} vs 10km Total Time')
    plt.tight_layout()
    return fig

#Generates an interactive chart of the mean progression across all checkpoints.
def plot_route_splits(df):

    # Calculate the mean time in minutes for each checkpoint
    means = {
        '2.5 km': df['split_2.5k_minutes'].mean(),
        '5.0 km': df['split_5k_minutes'].mean(),
        '7.5 km': df['split_7.5k_minutes'].mean(),
        '10.0 km': df['total_minutes'].mean()
    }

    distances = list(means.keys())
    times = list(means.values())

    fig = go.Figure()

    # The progression line
    fig.add_trace(go.Scatter(
        x=distances, y=times,
        mode='lines+markers',
        name='Mean Cumulative Time',
        line=dict(color='royalblue', width=4),
        marker=dict(size=12, color='darkblue'),
        hovertemplate='%{x} Checkpoint<br>Mean Time: %{y:.2f} mins<extra></extra>'
    ))

    # Adding Course Annotations based on the PDF context
    fig.add_annotation(x='2.5 km', y=times[0], text="Start Area:<br>Santiago Bernabéu", showarrow=True, arrowhead=2, ax=0, ay=-40)
    fig.add_annotation(x='7.5 km', y=times[2], text="7.5km Mark:<br>The Uphill 'Sting'", showarrow=True, arrowhead=2, ax=-30, ay=-50)
    fig.add_annotation(x='10.0 km', y=times[3], text="Finish Line:<br>Estadio de Vallecas", showarrow=True, arrowhead=2, ax=-40, ay=40)

    fig.update_layout(
        title="Interactive Race Progression & Elevation Impact",
        xaxis_title="Race Checkpoint",
        yaxis_title="Cumulative Mean Time (Minutes)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig