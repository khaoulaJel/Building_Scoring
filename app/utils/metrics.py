import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def calculate_average_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates average metrics for CO2 Usage, Water Usage, and Energy Consumption.

    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.

    Returns:
        dict: A dictionary of average metrics.
    """
    if df.empty:
        return {}

    return {
        "Average CO2 Usage": df["CO2_Usage"].mean(),
        "Average Water Usage": df["Water_Usage"].mean(),
        "Average Energy Consumption": df["Energy_Consumption"].mean(),
        "Most Common Class": df["class_label"].mode().iloc[0]
    }


def display_metrics_overview(df: pd.DataFrame) -> None:
    """
    Displays an overview of metrics in a Streamlit app.

    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
    """
    if df.empty:
        st.warning("No data to display metrics.")
        return

    st.subheader("Metrics Overview")
    metrics = calculate_average_metrics(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average CO₂ Usage", f"{metrics['Average CO2 Usage']:.1f} kg")
        st.metric("Average Water Usage", f"{metrics['Average Water Usage']:.0f} L")
    with col2:
        st.metric("Average Energy", f"{metrics['Average Energy Consumption']:.0f} kWh")
        st.metric("Class Distribution", f"{metrics['Most Common Class']} (most common)")


def display_benchmark_comparison(df: pd.DataFrame) -> None:
    """
    Displays a benchmark comparison for energy consumption, CO2 usage, and water usage.

    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
    """
    if df.empty:
        st.warning("No data to display benchmarks.")
        return

    st.subheader("Benchmark Comparison")

    benchmarks = {
        "Energy_Consumption": {"Excellent": 1000, "Good": 2000, "Average": 3000, "Poor": 4000},
        "CO2_Usage": {"Excellent": 100, "Good": 200, "Average": 300, "Poor": 400},
        "Water_Usage": {"Excellent": 2000, "Good": 4000, "Average": 6000, "Poor": 8000}
    }

    metric_for_benchmark = st.selectbox(
        "Select metric for benchmark comparison",
        ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    )

    avg_value = df[metric_for_benchmark].mean()

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=avg_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Average {metric_for_benchmark}"},
        gauge={
            'axis': {'range': [0, benchmarks[metric_for_benchmark]["Poor"] * 1.2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, benchmarks[metric_for_benchmark]["Excellent"]], 'color': 'green'},
                {'range': [benchmarks[metric_for_benchmark]["Excellent"], benchmarks[metric_for_benchmark]["Good"]], 'color': 'lightgreen'},
                {'range': [benchmarks[metric_for_benchmark]["Good"], benchmarks[metric_for_benchmark]["Average"]], 'color': 'yellow'},
                {'range': [benchmarks[metric_for_benchmark]["Average"], benchmarks[metric_for_benchmark]["Poor"]], 'color': 'orange'},
                {'range': [benchmarks[metric_for_benchmark]["Poor"], benchmarks[metric_for_benchmark]["Poor"] * 1.2], 'color': 'red'}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Performance interpretation
    if avg_value <= benchmarks[metric_for_benchmark]["Excellent"]:
        performance = "Excellent - Buildings are performing at top efficiency levels"
    elif avg_value <= benchmarks[metric_for_benchmark]["Good"]:
        performance = "Good - Buildings are performing well but have room for improvement"
    elif avg_value <= benchmarks[metric_for_benchmark]["Average"]:
        performance = "Average - Consider moderate efficiency improvements"
    elif avg_value <= benchmarks[metric_for_benchmark]["Poor"]:
        performance = "Below Average - Significant improvements recommended"
    else:
        performance = "Poor - Urgent efficiency measures required"

    st.write(f"**Performance Assessment:** {performance}")