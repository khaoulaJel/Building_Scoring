import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def export_to_csv(df: pd.DataFrame, filename: str = "building_data.csv") -> None:
    """
    Exports a DataFrame to a CSV file and provides a download button in Streamlit.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        filename (str): The name of the output CSV file.
    """
    if df.empty:
        st.warning("No data to export.")
        return

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )


def generate_report_summary(df: pd.DataFrame, filename: str = "building_report.md") -> None:
    """
    Generates a markdown report summary from a DataFrame and provides a download button in Streamlit.

    Args:
        df (pd.DataFrame): The DataFrame to generate the report from.
        filename (str): The name of the output markdown file.
    """
    if df.empty:
        st.warning("No data to generate a report.")
        return

    report = f"""# Building Analysis Report
Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Summary
- Total Buildings: {len(df)}
- Average Energy Consumption: {df['Energy_Consumption'].mean():.1f} kWh
- Average CO2 Usage: {df['CO2_Usage'].mean():.1f} kg
- Average Water Usage: {df['Water_Usage'].mean():.1f} L

## Class Distribution
{df['class_label'].value_counts().to_string()}
"""
    st.download_button(
        label="Download Report",
        data=report,
        file_name=filename,
        mime="text/markdown"
    )


def add_export_section(df: pd.DataFrame) -> None:
    """
    Adds an export section to the Streamlit app for exporting data and generating reports.

    Args:
        df (pd.DataFrame): The DataFrame to export or generate a report from.
    """
    st.subheader("Export Data")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to CSV"):
            export_to_csv(df)

    with col2:
        if st.button("Generate Report Summary"):
            generate_report_summary(df)

def add_benchmark_comparison(df: pd.DataFrame) -> None:
    """
    Displays a benchmark comparison for energy consumption, CO2 usage, and water usage.

    Args:
        df (pd.DataFrame): The DataFrame containing the metrics.
    """
    if df.empty:
        st.warning("No data to display benchmarks.")
        return

    st.subheader("Benchmark Comparison")

    # Define benchmark thresholds for each metric
    benchmarks = {
        "Energy_Consumption": {"Excellent": 1000, "Good": 2000, "Average": 3000, "Poor": 4000},
        "CO2_Usage": {"Excellent": 100, "Good": 200, "Average": 300, "Poor": 400},
        "Water_Usage": {"Excellent": 2000, "Good": 4000, "Average": 6000, "Poor": 8000}
    }

    # Let the user select which metric to compare
    metric_for_benchmark = st.selectbox(
        "Select metric for benchmark comparison",
        ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    )

    # Calculate the average value for the selected metric
    avg_value = df[metric_for_benchmark].mean()

    # Create a Plotly gauge chart to visualize the benchmark
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

    # Performance interpretation based on the average value
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