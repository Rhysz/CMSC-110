'''
BAGORIO, Ivan ; CAPULE, Gian ; DIZON, Ahrdy ; RIÑON, Cedric
2026 - 05
Description: EDA functions
'''

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def clean_data(filepath):
    df = pd.read_csv(filepath)

    df = df.dropna(subset=['2.5km_seconds', '5km_seconds', '7.5km_seconds', 'total_seconds', 'age_category', 'sex'])
    df = df.drop_duplicates()

    # Filter 1: Physically impossible timing errors (5K time >= Total time)
    df = df[df['total_seconds'] > df['5km_seconds']]

    # Calculate second-half time for advanced filtering
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

    df['split_2.5k_minutes'] = df['2.5km_seconds'] / 60
    df['split_5k_minutes'] = df['5km_seconds'] / 60
    df['split_7.5k_minutes'] = df['7.5km_seconds'] / 60
    df['total_minutes'] = df['total_seconds'] / 60

    df['sex'] = df['sex'].str.upper().str.strip().replace({'M': 'Male', 'F': 'Female'})

    return df


def plot_univariate(df, bins=30):
    min_val = df['total_seconds'].min()
    max_val = df['total_seconds'].max()
    tick_vals = np.linspace(min_val, max_val, 8)
    tick_text = [f"{int(v // 60)}:{int(v % 60):02d}" for v in tick_vals]

    df_plot = df.copy()
    df_plot['Formatted Time'] = df_plot['total_seconds'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}")

    fig = px.histogram(
        df_plot,
        x='total_seconds',
        nbins=bins,
        title='Univariate Analysis: Distribution of Overall Finish Times',
        color_discrete_sequence=['skyblue'],
        labels={'total_seconds': '10km Total Finish Time'},
        hover_data={'total_seconds': False, 'Formatted Time': True}
    )

    mean_val = df['total_seconds'].mean()
    mean_text = f"Mean: {int(mean_val//60)}:{int(mean_val%60):02d}"

    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=mean_text, annotation_position="top right")

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            title='10km Total Finish Time (MM:SS)'
        ),
        yaxis_title='Runner Count',
        template='plotly_white'
    )
    return fig


def plot_bivariate(df):
    order = sorted(df['age_category'].unique())

    df_plot = df.copy()
    df_plot['Formatted Time'] = df_plot['total_seconds'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}")

    fig = px.box(
        df_plot,
        x='age_category',
        y='total_seconds',
        color='sex',
        category_orders={'age_category': order},
        title='Bivariate Analysis: Finish Times Across Age and Gender',
        color_discrete_sequence=px.colors.qualitative.Pastel1,
        labels={'age_category': 'Age Category', 'sex': 'Gender'},
        hover_data={'total_seconds': False, 'Formatted Time': True}
    )

    min_val = df['total_seconds'].min()
    max_val = df['total_seconds'].max()
    tick_vals = np.linspace(min_val, max_val, 8)
    tick_text = [f"{int(v // 60)}:{int(v % 60):02d}" for v in tick_vals]

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            title='10km Total Finish Time (MM:SS)'
        ),
        xaxis_title='Age Category',
        legend_title='Gender',
        template='plotly_white'
    )
    return fig


def plot_efficiency(df):
    # Sort data to isolate the fastest and slowest runners
    df_sorted = df.sort_values('total_seconds')

    if len(df_sorted) > 1000:
        # Guarantee inclusion of extreme outliers (top 50 and bottom 50)
        top_runners = df_sorted.head(50)
        bottom_runners = df_sorted.tail(50)
        mid_runners = df_sorted.iloc[50:-50].sample(900)
        sample_df = pd.concat([top_runners, bottom_runners, mid_runners])
    else:
        sample_df = df_sorted.copy()

    sample_df['5km Time'] = sample_df['5km_seconds'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}")
    sample_df['Total Time'] = sample_df['total_seconds'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}")

    fig = px.scatter(
        sample_df,
        x='5km_seconds',
        y='total_seconds',
        color='split_strategy',
        color_discrete_map={
            'Negative Split (Faster 2nd Half)': 'green',
            'Positive Split (Slower 2nd Half)': 'red'
        },
        opacity=0.6,
        title='Multivariate Analysis: Pacing Efficiency (5km Split vs. Final Time)',
        labels={'split_strategy': 'Pacing Strategy'},
        hover_data={'5km_seconds': False, 'total_seconds': False, '5km Time': True, 'Total Time': True}
    )

    x_min, x_max = sample_df['5km_seconds'].min(), sample_df['5km_seconds'].max()
    y_min, y_max = sample_df['total_seconds'].min(), sample_df['total_seconds'].max()

    x_ticks = np.linspace(x_min, x_max, 6)
    y_ticks = np.linspace(y_min, y_max, 6)

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=[f"{int(v // 60)}:{int(v % 60):02d}" for v in x_ticks],
            title='5km Split Time (MM:SS)'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=y_ticks,
            ticktext=[f"{int(v // 60)}:{int(v % 60):02d}" for v in y_ticks],
            title='10km Total Finish Time (MM:SS)'
        ),
        legend_title='Pacing Strategy',
        template='plotly_white'
    )
    return fig


def plot_correlation(df, chosen_metric, chosen_label):
    display_df = df[[chosen_metric, 'total_seconds']].rename(
        columns={chosen_metric: chosen_label, 'total_seconds': '10km Total Time'}
    )

    corr_matrix = display_df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".4f",
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title=f'Correlation Matrix: {chosen_label} vs 10km Total Time'
    )

    fig.update_layout(template='plotly_white')
    return fig


# Generates an interactive chart of the mean progression vs the fastest runner
def plot_route_splits(df):
    # Calculate the mean time in minutes for each checkpoint
    means = {
        '2.5 km': df['split_2.5k_minutes'].mean(),
        '5.0 km': df['split_5k_minutes'].mean(),
        '7.5 km': df['split_7.5k_minutes'].mean(),
        '10.0 km': df['total_minutes'].mean()
    }

    # Extract the exact splits of the fastest runner in the filtered data
    fastest_runner = df.loc[df['total_minutes'].idxmin()]
    fastest = {
        '2.5 km': fastest_runner['split_2.5k_minutes'],
        '5.0 km': fastest_runner['split_5k_minutes'],
        '7.5 km': fastest_runner['split_7.5k_minutes'],
        '10.0 km': fastest_runner['total_minutes']
    }

    distances = list(means.keys())
    mean_times = list(means.values())
    fastest_times = list(fastest.values())

    fig = go.Figure()

    # The mean progression line
    fig.add_trace(go.Scatter(
        x=distances, y=mean_times,
        mode='lines+markers',
        name='Mean Cumulative Time',
        line=dict(color='royalblue', width=4),
        marker=dict(size=12, color='darkblue'),
        hovertemplate='%{x} Checkpoint<br>Mean Time: %{y:.2f} mins<extra></extra>'
    ))

    # The fastest runner progression line
    fig.add_trace(go.Scatter(
        x=distances, y=fastest_times,
        mode='lines+markers',
        name='Fastest Runner',
        line=dict(color='gold', width=4, dash='dash'),
        marker=dict(size=12, color='darkgoldenrod', symbol='star'),
        hovertemplate='%{x} Checkpoint<br>Fastest Time: %{y:.2f} mins<extra></extra>'
    ))

    # Adding Course Annotations based on the PDF context
    fig.add_annotation(x='2.5 km', y=mean_times[0], text="Start Area:<br>Santiago Bernabéu", showarrow=True, arrowhead=2, ax=0, ay=-40)
    fig.add_annotation(x='7.5 km', y=mean_times[2], text="7.5km Mark:<br>The Uphill 'Sting'", showarrow=True, arrowhead=2, ax=-30, ay=-50)
    fig.add_annotation(x='10.0 km', y=mean_times[3], text="Finish Line:<br>Estadio de Vallecas", showarrow=True, arrowhead=2, ax=-40, ay=40)

    fig.update_layout(
        title="Interactive Race Progression & Elevation Impact",
        xaxis_title="Race Checkpoint",
        yaxis_title="Cumulative Time (Minutes)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white'
    )

    return fig