"""
This module provides functions for creating various visualizations using mostly Plotly.
It includes functions for plotting confidence intervals, proportions, and so on.
The functions are designed to produce charts that can be displayed in
Jupyter notebooks or saved as image files.
"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IPython.display import Image, display
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import shap

from utils.stats_utils import compute_confidence_intervals


PLOTLY_MODES = ["github", "notebook"]
PLOTLY_MODE = PLOTLY_MODES[0]

PASTEL_COLORS_RGB = {
    "aqua blue": "rgb(102, 197, 204)",
    "coral orange": "rgb(248, 156, 116)",
    "neutral grey": "rgb(179, 179, 179)",
    "leafy green": "rgb(135, 197, 95)",
    "sunny yellow": "rgb(246, 207, 113)",
    "soft purple": "rgb(220, 176, 242)",
    "lavender blue": "rgb(158, 185, 243)",
    "bubblegum pink": "rgb(254, 136, 177)",
    "lime zest": "rgb(201, 219, 116)",
    "mint green": "rgb(139, 224, 164)",
    "purple blue": "rgb(180, 151, 231)",
}
pastel_colors_list = list(PASTEL_COLORS_RGB.values())

pio.templates["plotly_dark"].layout.colorway = pastel_colors_list
pio.templates.default = "plotly_dark"

aqua_blue_grayish_colorscale = [
    [0, "rgb(224, 224, 224)"],
    [0.25, "rgb(102, 170, 170)"],
    [0.5, "rgb(76, 140, 140)"],
    [0.75, "rgb(51, 110, 110)"],
    [1.0, "rgb(25, 80, 80)"],
]


def plot_horizontal_bars(
    df: pd.DataFrame,
    feature_column: str,
    values_column: str,
    custom_title: str = "Stroke Prediction Dataset Features Ranked",
    custom_subtitle: str = "by Mutual Information Score",
    image_name: str = "horizontal_barchart",
    width_px: int = 506,
    height_px: int = 344,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Plots sorted horizontal bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be plotted.
    entity_column : str
        The column name representing the feature.
    values_column : str
        The column name representing the numeric feature used for sorting and bar
        length.
    custom_title : str
        Custom title for the plot.
    custom_subtitle : str
        Custom subtitle for the plot.
    image_name : str
        File name for saving the plot image if `plotly_mode` is not "notebook".
    width_px : int
        Width of the plot in pixels.
    height_px : int
        Height of the plot in pixels.
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot
        is displayed inline. Otherwise, the plot is saved as an image.
    """
    df = df.sort_values(by=values_column, ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df[feature_column],
            x=df[values_column],
            orientation="h",
        )
    )

    fig.update_layout(
        width=width_px,
        height=height_px,
        xaxis=dict(side="top"),
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.94,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
        margin=dict(l=10, r=30, t=90, b=60),
    )
    fig.update_yaxes(
        ticks="outside",
        tickcolor="black",
        ticklen=10,
    )
    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}.png")
        display(Image(f"plotly_charts/{image_name}.png"))


def plot_category_counts(
    df: pd.DataFrame,
    category_column: str,
    custom_title: str = "Ranked Categories",
    custom_subtitle: str = "by count of entities per category",
    image_name: str = "category_counts",
    width_px: int = 776,
    height_px: int = 384,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Plots a bar chart showing the counts of entities in each category.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be analyzed.
    category_column : str
        The column name representing the categories to count.
    custom_title : str
        Title of the bar chart.
    custom_subtitle : str
        Subtitle of the bar chart.
    image_name : str
        File name for saving the plot image if `plotly_mode` is not "notebook".
    width_px : int
        Width of the plot in pixels.
    height_px : int
        Height of the plot in pixels.
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot
        is displayed inline. Otherwise, the plot is saved as an image.
    """
    category_counts = df[category_column].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                text=category_counts.values,
                textposition="outside",
                hovertemplate="%{y} courses organized by %{x}",
            )
        ]
    )

    fig.update_layout(
        width=width_px,
        height=height_px,
        yaxis=dict(range=[0, category_counts.max() * 1.2]),
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.9,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}.png")
        display(Image(f"plotly_charts/{image_name}.png"))


def plot_histogram_with_hue(
    df: pd.DataFrame,
    target_metric_column: str,
    hue_column: str,
    bins=20,
    custom_title: str = "Histogram",
    custom_subtitle: str = "Number of observations per bin",
    image_name: str = "distribution_histogram",
    width_px: int = 800,
    height_px: int = 484,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Creates a histogram for a specified target metric, differentiated by hues based on
    another column.

    Parameters
    ----------
    df : pd.DataFrame
        The input data as a pandas DataFrame.
    target_metric_column : str
        The column whose distribution is to be plotted.
    hue_column : str
        The column used to differentiate data within the histogram (e.g., by color).
    bins : int, optional
        The number of bins to use in the histogram (default is 20).
    custom_title : str, optional
        The main title for the plot (default is "Histogram").
    custom_subtitle : str, optional
        The subtitle for the plot (default is "Number of observations per bin").
    image_name : str, optional
        The name of the output image file if saving the plot.
    width_px : int, optional
        The width of the histogram in pixels (default is 800).
    height_px : int, optional
        The height of the plot in pixels (default is 484).
    plotly_mode : str, optional
        The mode to render the plot: "notebook" to display inline, or else to write
        to a file.
    """
    hue_values = np.sort(df[hue_column].unique())

    fig = go.Figure()
    for hue_value in hue_values:
        subset = df[df[hue_column] == hue_value]
        fig.add_trace(
            go.Histogram(
                x=subset[target_metric_column],
                nbinsx=bins,
                name=str(hue_value),
                opacity=0.5,
            )
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
        barmode="overlay",
        showlegend=True,
        legend_title=hue_column.replace("_", " ").title(),
    )
    fig.update_xaxes(title_text="")

    fig.add_annotation(
        x=1.0,
        y=-0.15,
        text=target_metric_column.replace("_", " ").title(),
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}_{hue_column}.png")
        display(Image(f"plotly_charts/{image_name}_{hue_column}.png"))


def plot_qq_with_histogram(
    df: pd.DataFrame,
    column: str,
    custom_title: str = "QQ Plot and Histogram",
    custom_hist_y_label="Number of observations",
    custom_qq_y_label="Sample quantiles",
    bins: int = 30,
    width_px: int = 1000,
    height_px: int = 450,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Creates a QQ plot (to check normality) and a histogram (to visualize distribution)
    for a specified numerical column. The histogram includes vertical lines for mean,
    median, 25th & 75th percentiles, and IQR-based lower and upper fences.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the column to analyze.
    column : str
        The numerical column to generate QQ plot and histogram for.
    custom_title : str, optional
        Title for the overall figure, by default "QQ Plot and Histogram".
    custom_hist_y_label : str, optional
        Y-axis label for the histogram, by default "Number of observations".
    custom_qq_y_label : str, optional
        Y-axis label for the QQ plot, by default "Sample quantiles".
    width_px : int, optional
        Width of the figure in pixels, by default 900.
    height_px : int, optional
        Height of the figure in pixels, by default 450.
    bins : int, optional
        Number of bins for the histogram, by default 30.
    plotly_mode : str, optional
        Mode to determine interactive or static display, by default "notebook".
    """
    feature_data = df[column].dropna()

    feature_mean = feature_data.mean()
    feature_median = feature_data.median()
    q25 = feature_data.quantile(0.25)
    q75 = feature_data.quantile(0.75)
    iqr = q75 - q25
    lower_fence = q25 - 1.5 * iqr
    upper_fence = q75 + 1.5 * iqr

    feature_std = feature_data.std()
    standardized_data = (feature_data - feature_mean) / feature_std

    qq = sm.ProbPlot(standardized_data, dist=norm)
    theoretical_quantiles = qq.theoretical_quantiles * feature_std + feature_mean
    sample_quantiles = qq.sample_quantiles * feature_std + feature_mean

    min_val = min(min(sample_quantiles), min(theoretical_quantiles))
    max_val = max(max(sample_quantiles), max(theoretical_quantiles))

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        shared_yaxes=False,
    )

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            name="Sample Quantiles",
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="45-degree Line",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=feature_data,
            nbinsx=bins,
            marker=dict(opacity=0.7),
            name="Count",
        ),
        row=1,
        col=2,
    )

    line_styles = {
        "mean": dict(color=PASTEL_COLORS_RGB["lavender blue"], dash="solid"),
        "median": dict(color=PASTEL_COLORS_RGB["lavender blue"], dash="dash"),
        "q25": dict(color=PASTEL_COLORS_RGB["leafy green"], dash="dot"),
        "q75": dict(color=PASTEL_COLORS_RGB["leafy green"], dash="dot"),
        "lower_fence": dict(color=PASTEL_COLORS_RGB["soft purple"], dash="dot"),
        "upper_fence": dict(color=PASTEL_COLORS_RGB["soft purple"], dash="dot"),
    }

    for name, value in [
        ("mean", feature_mean),
        ("median", feature_median),
        ("q25", q25),
        ("q75", q75),
        ("lower_fence", lower_fence),
        ("upper_fence", upper_fence),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[value, value],
                y=[0, max(np.histogram(feature_data, bins=bins)[0]) * 1.2],
                mode="lines",
                name=name.replace("_", " ").title(),
                line=line_styles[name],
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=True,
        margin=dict(l=30, r=30, t=70, b=60),
    )

    fig.add_annotation(
        x=-0.03,
        y=1.07,
        text=custom_qq_y_label,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="left",
    )

    fig.add_annotation(
        x=0.42,
        y=1.07,
        text=custom_hist_y_label,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="left",
    )

    fig.add_annotation(
        x=1.0,
        y=-0.15,
        text=column.replace("_", " ").title(),
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    fig.add_annotation(
        x=0.36,
        y=-0.15,
        text="Theoretical Quantiles",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        image_path = f"plotly_charts/qq_hist_{column}.png"
        pio.write_image(fig, image_path)
        display(Image(image_path))


def highlight_proportions(val: float) -> str:
    """
    This function assigns a background color to a numerical proportion
    based on predefined color thresholds from the `aqua_blue_grayish_colorscale`
    list. The color intensity varies according to the value range.

    Parameters
    ----------
    val : float
        The numerical proportion value, expected to be in the range [0, 1].

    Returns
    -------
    str
        A CSS style string specifying the background color.
    """
    if val > 0.8:
        color = aqua_blue_grayish_colorscale[4][1]
    elif val > 0.6:
        color = aqua_blue_grayish_colorscale[3][1]
    elif val > 0.4:
        color = aqua_blue_grayish_colorscale[2][1]
    elif val > 0.2:
        color = aqua_blue_grayish_colorscale[1][1]
    else:
        color = aqua_blue_grayish_colorscale[0][1]

    return f"background-color: {color}"


def plot_heatmap(
    df: pd.DataFrame,
    custom_title: str = "Feature 1 vs. feature 2 relationship",
    custom_subtitle: str = "by aggregated value",
    image_name: str = "heatmap",
    width_px: int = 576,
    height_px: int = 484,
    legend_title: str = "",
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Creates a heatmap visualization from a DataFrame. The heatmap
    displays values with a custom color scale and hover information.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the values to plot in the heatmap. The rows and columns
        should correspond to the features being compared, and the values should
        represent the relationship (e.g., p-values, correlations) between them.
    custom_title : str, optional
        The main title for the plot (default is "Feature 1 vs. feature 2 relationship").
    custom_subtitle : str, optional
        The subtitle for the plot (default is "by aggregated value").
    image_name : str, optional
        The name of the output image file if saving the plot (default is "heatmap").
    width_px : int, optional
        The width of the plot in pixels (default is 576).
    height_px : int, optional
        The height of the plot in pixels (default is 484).
    plotly_mode : str, optional
        The mode to render the plot: "notebook" to display inline, or else to write
        to a file (default is PLOTLY_MODE).
    """

    heatmap_data = df.values
    groups = df.columns.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=groups,
            y=groups,
            colorscale=aqua_blue_grayish_colorscale,
            colorbar=dict(title=legend_title),
            zmin=-1,
            zmax=1,
            text=heatmap_data,
            texttemplate="%{text:.1f}",
            hovertemplate="%{x} vs. %{y}<br>P-value: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.9,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
    )
    fig.update_xaxes(ticklabelstandoff=10)
    fig.update_yaxes(ticklabelstandoff=10)

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}.png")
        display(Image(f"plotly_charts/{image_name}.png"))


def plot_violin(
    df: pd.DataFrame,
    numeric_column: str,
    custom_text: str = "Distribution and Density",
    custom_subtitle: str = "of feature",
    image_name: str = "violin_distribution",
    width_px: int = 576,
    height_px: int = 384,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Plots violin plots for numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame containing the data to be plotted.
    variables_df : pd.DataFrame
        A DataFrame containing metadata about the columns in `df`, including a
        "typical_range" column used for grouping.
    numeric_column : str
        A numeric column name from `df` to be plotted.
    custom_text : str, optional
        Custom title text for the plot.
    custom_subtitle : str, optional
        Custom subtitle text for the plot.
    image_name : str, optional
        Base name for saving image files when `plotly_mode` is not set to "notebook".
    width_px : int, optional
        Width of the plot in pixels.
    height_px : int, optional
        Height of the plot in pixels.
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot
        is displayed inline. Otherwise, the plot is saved as an image.
    """

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=df[numeric_column],
            name=numeric_column.replace("_", " ").capitalize(),
            box_visible=True,
            points="all",
        )
    )
    fig.update_layout(
        width=width_px,
        height=height_px,
        xaxis=dict(side="bottom", ticktext=[], tickvals=[]),
        title=dict(
            text=custom_text,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.9,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
    ),

    if plotly_mode == "notebook":
        fig.show()
    else:
        file_name = f"plotly_charts/{image_name}_{numeric_column}.png"
        pio.write_image(fig, file_name)
        display(Image(file_name))


def plot_target_col_ratio_ci_distribution(
    df: pd.DataFrame,
    input_col: str,
    target_col: str,
    y_threshold: float = 0.5,
    custom_title: str = "Target Feature Group of Interest by Feature",
    scatter_title: str = "Group of Interest proportion with 95% CI",
    secondary_chart_title: str = "Distribution by Target Feature",
    custom_x_axis_title: str = "Feature",
    image_name: str = "proportion_scatter_ci",
    width_px: int = 1100,
    height_px: int = 450,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Generates a visualization of the relationship between a categorical or numerical
    feature and the target variable (`target_col`) using a scatter plot with
    confidence intervals and a distribution plot.

    The function calculates the proportion of customers who purchased travel insurance
    for each unique value of the `input_col`, along with its 95% confidence interval.
    Additionally, it visualizes the distribution of the feature with respect to the
    target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the feature and target variable.
    input_col : str
        The feature column to analyze.
    target_col : str
        The target variable (binary: 0 or 1) indicating if a customer bought travel
        insurance.
    y_threshold : float, optional
        The maximum value for the y-axis, by default 0.5.
    custom_title : str, optional
        The main title for the plot, by default "Customers Buying Travel Insurance
        by Feature".
    scatter_title : str, optional
        The title for the scatter plot showing proportions with confidence intervals,
        by default "Buying proportion with 95% CI".
    secondary_chart_title : str, optional
        The title for the secondary chart showing distribution by insurance purchase
        status, by default "Distribution by Insurance Purchase Status".
    custom_x_axis_title : str, optional
        The x-axis title, by default "Feature".
    image_name : str, optional
        The filename to save the image when using static mode, by default
        "proportion_scatter_ci".
    width_px : int, optional
        The width of the figure in pixels, by default 1100.
    height_px : int, optional
        The height of the figure in pixels, by default 450.
    plotly_mode : str, optional
        Determines whether to display the plot in a notebook ("notebook") or save it
        as an image, by default "notebook".
    """
    df_grouped = (
        df.groupby([input_col], observed=True)[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    df_grouped["ratio"] = df_grouped["sum"] / df_grouped["count"]

    df_grouped["lower_ci"], df_grouped["upper_ci"] = zip(
        *df_grouped.apply(
            lambda row: compute_confidence_intervals(row["sum"], row["count"]), axis=1
        )
    )

    unique_values = sorted(df_grouped[input_col].unique())
    is_binary = unique_values == [0, 1]
    is_categorical = df[input_col].dtype in ["object", "category"]
    is_numeric = df[input_col].dtype in ["int64", "float64"]

    xaxis_settings = {}
    if is_binary:
        df_grouped[input_col] = df_grouped[input_col].replace({0: "No", 1: "Yes"})
        x_tickvals = list(df_grouped[input_col].unique())
        xaxis_settings["tickvals"] = x_tickvals
        xaxis_settings["ticktext"] = x_tickvals

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[scatter_title, secondary_chart_title],
        column_widths=[0.5, 0.5],
        shared_yaxes=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_grouped[input_col],
            y=df_grouped["ratio"],
            mode="markers+lines",
            name="Proportion",
            marker=dict(size=15),
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_grouped["upper_ci"] - df_grouped["ratio"],
                arrayminus=df_grouped["ratio"] - df_grouped["lower_ci"],
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if is_binary or is_categorical:
        df_bar = (
            df.groupby([input_col, target_col], observed=True)
            .size()
            .reset_index(name="count")
        )
        if is_binary:
            df_bar[input_col] = df_bar[input_col].replace({0: "No", 1: "Yes"})

        for t_value in df_bar[target_col].unique():
            fig.add_trace(
                go.Bar(
                    x=df_bar[df_bar[target_col] == t_value][input_col],
                    y=df_bar[df_bar[target_col] == t_value]["count"],
                    name="Yes" if t_value == 1 else "No",
                ),
                row=1,
                col=2,
            )

    elif is_numeric:
        for t_value in df[target_col].unique():
            fig.add_trace(
                go.Histogram(
                    x=df[df[target_col] == t_value][input_col],
                    name="Yes" if t_value == 1 else "No",
                    opacity=0.7,
                    bingroup="overlay",
                    histnorm=None,
                ),
                row=1,
                col=2,
            )
        fig.update_traces(opacity=0.75)

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            x=0.5,
            font=dict(size=20),
        ),
        margin=dict(l=30, r=30, t=70, b=60),
        barmode="group" if is_binary or is_categorical else "overlay",
        showlegend=True,
        legend=dict(
            title="Stroke:",
        ),
    )
    fig.update_yaxes(range=[0, y_threshold], tickformat=".0%", row=1, col=1)

    fig.update_xaxes(title_text="")
    fig.add_annotation(
        x=1.0,
        y=-0.15,
        text=custom_x_axis_title,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}_{input_col}.png")
        display(Image(f"plotly_charts/{image_name}_{input_col}.png"))


def plot_feature_proportion_and_distribution(
    df: pd.DataFrame,
    input_col: str,
    target_col: str,
    hue_column: str,
    custom_title: str = "",
    scatter_title: str = "Buying proportion with 95% CI",
    secondary_chart_title: str = "Customers Count that Bought Insurance",
    custom_x_axis_title: str = "Feature",
    image_name: str = "proportion_scatter_ci",
    width_px: int = 1100,
    height_px: int = 450,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Generates visualizations comparing the proportion of the target variable
    across different values of a feature, with an additional grouping factor.

    This function creates two subplots:
    - A scatter plot showing the proportion of the target variable (`target_col`)
      across values of `input_col`, with confidence intervals.
    - A bar chart showing the distribution of the feature by insurance purchase status.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the feature and target variable.
    input_col : str
        The feature column to analyze.
    target_col : str
        The target variable (binary: 0 or 1) indicating if a customer bought travel
        insurance.
    hue_column : str
        The column used for grouping (e.g., gender, employment type).
    custom_title : str, optional
        The main title for the plot, by default "".
    scatter_title : str, optional
        The title for the scatter plot showing proportions with confidence intervals,
        by default "Buying proportion with 95% CI".
    secondary_chart_title : str, optional
        The title for the secondary chart showing the count of customers who bought
        insurance, by default "Customers Count that Bought Insurance".
    custom_x_axis_title : str, optional
        The x-axis title, by default "Feature".
    image_name : str, optional
        The filename to save the image when using static mode, by default
        "proportion_scatter_ci".
    width_px : int, optional
        The width of the figure in pixels, by default 1100.
    height_px : int, optional
        The height of the figure in pixels, by default 450.
    plotly_mode : str, optional
        Determines whether to display the plot in a notebook ("notebook")
        or save it as an image.
    """
    df = df.copy()
    df_grouped = (
        df.groupby([input_col, hue_column])[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    df_grouped["ratio"] = df_grouped["sum"] / df_grouped["count"]

    df_grouped["lower_ci"], df_grouped["upper_ci"] = zip(
        *df_grouped.apply(
            lambda row: compute_confidence_intervals(row["sum"], row["count"]), axis=1
        )
    )

    unique_values = sorted(df[input_col].unique())
    is_binary = unique_values == [0, 1]

    xaxis_settings = {}
    if is_binary:
        df_grouped[input_col] = df_grouped[input_col].replace({0: "No", 1: "Yes"})
        df[input_col] = df[input_col].replace({0: "No", 1: "Yes"})
        x_tickvals = list(df_grouped[input_col].unique())
        xaxis_settings["tickvals"] = x_tickvals
        xaxis_settings["ticktext"] = x_tickvals

    unique_hue_values = df[hue_column].unique()
    color_map = {
        hue_value: pastel_colors_list[i % len(pastel_colors_list)]
        for i, hue_value in enumerate(unique_hue_values)
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[scatter_title, secondary_chart_title],
        column_widths=[0.5, 0.5],
        shared_yaxes=False,
    )

    for hue_value in df_grouped[hue_column].unique():
        subset = df_grouped[df_grouped[hue_column] == hue_value]
        fig.add_trace(
            go.Scatter(
                x=subset[input_col],
                y=subset["ratio"],
                mode="markers+lines",
                name="Yes" if hue_value == 1 else "No",
                marker=dict(size=10, color=color_map[hue_value]),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=subset["upper_ci"] - subset["ratio"],
                    arrayminus=subset["ratio"] - subset["lower_ci"],
                ),
                legendgroup=f"group_{hue_value}",
            ),
            row=1,
            col=1,
        )

    df_bar = (
        df[df[target_col] == 1]
        .groupby([input_col, hue_column])
        .size()
        .reset_index(name="count")
    )

    if is_binary:
        df_bar[input_col] = df_bar[input_col].replace({0: "No", 1: "Yes"})

    for hue_value in df_bar[hue_column].unique():
        fig.add_trace(
            go.Bar(
                x=df_bar[df_bar[hue_column] == hue_value][input_col],
                y=df_bar[df_bar[hue_column] == hue_value]["count"],
                name="Yes" if hue_value == 1 else "No",
                marker=dict(color=color_map[hue_value]),
                legendgroup=f"group_{hue_value}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        margin=dict(l=30, r=30, t=70, b=60),
        barmode="group",
        showlegend=True,
        legend=dict(
            title=hue_column.replace("_", " ").title(),
        ),
    )

    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="")

    fig.add_annotation(
        x=1.0,
        y=-0.15,
        text=custom_x_axis_title,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}_{input_col}_{hue_column}.png")
        display(Image(f"plotly_charts/{image_name}_{input_col}_{hue_column}.png"))


def plot_double_sided_violin(
    df: pd.DataFrame,
    categorical_col: str,
    numerical_col: str,
    target_col: str,
    custom_title: str = "Double-Sided Violin Plot",
    custom_x_axis_title: str = "Feature",
    width_px: int = 820,
    height_px: int = 500,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Creates a double-sided violin plot to visualize the distribution of a numerical
    variable split by a categorical feature and grouped by a binary target variable.

    This function generates a **double-sided violin plot**, where the distributions of
    the numerical feature (`numerical_col`) are plotted separately for each category
    in `categorical_col`, with respect to the target variable (`target_col`).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the feature and target variable.
    categorical_col : str
        The categorical feature to group the distributions.
    numerical_col : str
        The numerical variable whose distribution is plotted.
    target_col : str
        The binary target variable (0 or 1) used to split the violin plots.
    custom_title : str, optional
        The main title for the plot, by default "Double-Sided Violin Plot".
    custom_x_axis_title : str, optional
        The x-axis title, by default "Feature".
    width_px : int, optional
        The width of the figure in pixels, by default 820.
    height_px : int, optional
        The height of the figure in pixels, by default 500.
    plotly_mode : str, optional
        Determines whether to display the plot in a notebook ("notebook")
        or save it as an image.
    """
    unique_hue_values = df[target_col].unique()
    color_map = {
        hue_value: pastel_colors_list[i % len(pastel_colors_list) + 1]
        for i, hue_value in enumerate(unique_hue_values)
    }

    fig = go.Figure()

    for hue_value in unique_hue_values:
        subset = df[df[target_col] == hue_value]
        fig.add_trace(
            go.Violin(
                x=subset[categorical_col],
                y=subset[numerical_col],
                name="Yes" if hue_value == 1 else "No",
                side="positive" if hue_value == 1 else "negative",
                line_color=color_map[hue_value],
                fillcolor=color_map[hue_value],
                opacity=0.7,
                points="all",
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(
            text=numerical_col.replace("_", " ").capitalize(), font=dict(size=14)
        ),
        xaxis_title=custom_x_axis_title,
        margin=dict(l=50, r=50, t=70, b=60),
        showlegend=True,
        legend=dict(
            title="Purchased travel insurance:",
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            itemsizing="constant",
            font=dict(size=12),
        ),
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(
            fig,
            f"plotly_charts/violin_{categorical_col}_{numerical_col}_{target_col}.png",
        )
        display(
            Image(
                (
                    f"plotly_charts/violin_{categorical_col}_{numerical_col}_"
                    f"{target_col}.png"
                )
            )
        )


def plot_numerical_scatter_with_hue(
    df: pd.DataFrame,
    input_col_x: str,
    input_col_y: str,
    target_col: str,
    custom_title: str = "Scatter Plot of Numerical Features",
    width_px: int = 800,
    height_px: int = 500,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Creates a scatter plot to visualize the relationship between two numerical features,
    with points colored by a binary target variable.

    This function generates a scatter plot, where the relationship between two numerical
    columns (`input_col_x` and `input_col_y`) is plotted and colored by the binary
    target variable (`target_col`). It helps in visualizing trends and clusters based
    on the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the numerical features and target variable.
    input_col_x : str
        The first numerical feature to be plotted on the x-axis.
    input_col_y : str
        The second numerical feature to be plotted on the y-axis.
    target_col : str
        The binary target variable (0 or 1) used to color the points.
    custom_title : str, optional
        The main title for the plot, by default "Scatter Plot of Numerical Features".
    width_px : int, optional
        The width of the figure in pixels, by default 800.
    height_px : int, optional
        The height of the figure in pixels, by default 500.
    plotly_mode : str, optional
        Determines whether to display the plot in a notebook ("notebook") or
        save it as an image.
    """

    unique_hue_values = df[target_col].unique()
    color_map = {
        hue_value: pastel_colors_list[i % len(pastel_colors_list) + 1]
        for i, hue_value in enumerate(unique_hue_values)
    }

    fig = go.Figure()
    for hue_value in unique_hue_values:
        subset = df[df[target_col] == hue_value]
        fig.add_trace(
            go.Scatter(
                x=subset[input_col_x],
                y=subset[input_col_y],
                mode="markers",
                name="Yes" if hue_value == 1 else "No",
                marker=dict(size=10, color=color_map[hue_value]),
            )
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(
            text=input_col_y.replace("_", " ").capitalize(), font=dict(size=14)
        ),
        margin=dict(l=50, r=50, t=90, b=60),
        showlegend=True,
        legend=dict(
            title="Purchased travel insurance:",
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            itemsizing="constant",
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(title_text="")
    fig.add_annotation(
        x=1.0,
        y=-0.15,
        text=input_col_x.replace("_", " ").title(),
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
        xanchor="right",
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(
            fig, f"plotly_charts/scatter_{input_col_x}_{input_col_y}_{target_col}.png"
        )
        display(
            Image(f"plotly_charts/scatter_{input_col_x}_{input_col_y}_{target_col}.png")
        )


def plot_cv_boxplot(
    cv_fold_accuracies: dict,
    results_df: pd.DataFrame,
    custom_text: str = "Cross-Validation Performance",
    custom_subtitle: str = "Accuracy Across Folds",
    image_name: str = "cv_boxplot",
    width_px: int = 900,
    height_px: int = 600,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Plots a boxplot for cross-validation accuracies for different models.

    Parameters
    ----------
    cv_fold_accuracies : dict
        Dictionary where keys are model names and values are lists of accuracies for
        each fold.
    results_df : pd.DataFrame
        DataFrame containing model names in sorted order for display.
    custom_text : str, optional
        Custom title text for the plot.
    custom_subtitle : str, optional
        Custom subtitle text for the plot.
    image_name : str, optional
        Base name for saving image files when `plotly_mode` is not set to "notebook".
    width_px : int, optional
        Width of the plot in pixels.
    height_px : int, optional
        Height of the plot in pixels.
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot is
        displayed inline. Otherwise, the plot is saved as an image.
    """

    fig = go.Figure()
    for model_name in results_df["model"]:
        fig.add_trace(
            go.Box(
                y=cv_fold_accuracies[model_name],
                name=model_name,
                boxmean="sd",
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_text,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
        showlegend=False,
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        file_name = f"plotly_charts/{image_name}.png"
        pio.write_image(fig, file_name)
        display(Image(file_name))


def plot_confusion_matrices(
    confusion_matrices: dict,
    models_per_row: int = 3,
    width_px: int = 900,
    height_px: int = 600,
    custom_title: str = "Confusion Matrices for Models",
    image_name: str = "confusion_matrices",
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Visualizes confusion matrices for different models using Plotly Heatmap, displayed
    in a grid.

    Parameters
    ----------
    confusion_matrices : dict
        Dictionary where keys are model names and values are confusion matrix arrays.
    models_per_row : int, optional
        Number of confusion matrices to display per row.
    width_px : int, optional
        Total width of the plot in pixels.
    height_px : int, optional
        Total height of the plot in pixels.
    custom_title : str, optional
        Custom title text for the plot.
    image_name : str, optional
        Base name for saving image files when `plotly_mode` is not set to "notebook".
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot is
        displayed inline.
    """
    num_models = len(confusion_matrices)
    rows = (num_models + models_per_row - 1) // models_per_row

    fig = make_subplots(
        rows=rows,
        cols=models_per_row,
        subplot_titles=list(confusion_matrices.keys()),
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        row = idx // models_per_row + 1
        col = idx % models_per_row + 1

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                hoverinfo="z",
                coloraxis="coloraxis",
                text=cm,
                texttemplate="%{text}",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        width=width_px,
        height=height_px * rows,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.98,
            x=0.02,
            font=dict(size=20),
        ),
        margin=dict(l=30, r=30, t=80, b=60),
        coloraxis=dict(colorscale=aqua_blue_grayish_colorscale),
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        file_name = f"plotly_charts/{image_name}.png"
        pio.write_image(fig, file_name)
        display(Image(file_name))


def plot_bias_variance_scatter(
    df: pd.DataFrame,
    custom_title: str = "Bias - Variance Tradeoff",
    custom_subtitle: str = "The higher the bias, the lower the variance",
    image_name: str = "bias_variance_tradeoff",
    width_px: int = 650,
    height_px: int = 484,
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Plots a bias-variance tradeoff scatter plot using Plotly.

    This function creates a scatter plot with lines connecting the data points
    for bias squared and variance across different models. It visualizes the
    tradeoff between bias and variance, which is crucial in understanding
    model generalization.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing model names, bias squared values, and variance values.
        It must contain the columns `["model", "bias_squared", "variance"]`.
    custom_title : str, optional
        Title for the plot, by default "Bias-Variance Tradeoff".
    custom_subtitle : str, optional
        Subtitle for the plot, by default "The higher the bias, the lower the variance".
    image_name : str, optional
        File name for saving the plot image (only if `plotly_mode` is not "notebook"),
        by default "bias_variance_tradeoff".
    width_px : int, optional
        Width of the plot in pixels, by default 650.
    height_px : int, optional
        Height of the plot in pixels, by default 484.
    plotly_mode : str, optional
        Determines how the plot is displayed:
        - `"notebook"`: Displays inline in a Jupyter Notebook.
        - Any other value: Saves the plot as an image in "plotly_charts/" directory.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["model"],
            y=df["bias_squared"],
            mode="markers+lines",
            name="Bias^2",
            marker=dict(
                size=10,
                color=pastel_colors_list[1],
            ),
            line=dict(color=pastel_colors_list[1]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["model"],
            y=df["variance"],
            mode="markers+lines",
            name="Variance",
            marker=dict(size=10, color=pastel_colors_list[0]),
            line=dict(color=pastel_colors_list[0]),
        )
    )
    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.93,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
        xaxis=dict(
            tickangle=-45,
        ),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="center",
            x=0.8,
        ),
        margin=dict(l=40, r=40, t=80, b=100),
    )
    if plotly_mode == "notebook":
        fig.show()
    else:
        pio.write_image(fig, f"plotly_charts/{image_name}.png")
        display(Image(f"plotly_charts/{image_name}.png"))


def plot_feature_importances(
    df: pd.DataFrame,
    features_per_row: int = 2,
    width_px: int = 900,
    height_px: int = 300,
    custom_title: str = "Feature Importances for Models",
    image_name: str = "feature_importances",
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Visualizes feature importances for different models using horizontal bar charts,
    displayed in a grid.

    Parameters
    ----------
    merged_feature_importances_df : pd.DataFrame
        DataFrame where rows are models and columns are feature importances.
    features_per_row : int, optional
        Number of bar charts to display per row.
    width_px : int, optional
        Total width of the plot in pixels.
    height_px : int, optional
        Total height of the plot in pixels.
    custom_title : str, optional
        Custom title text for the plot.
    image_name : str, optional
        Base name for saving image files when `plotly_mode` is not set to "notebook".
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot
        is displayed inline, otherwise saved as an image.
    """
    num_models = len(df)
    rows = (num_models + features_per_row - 1) // features_per_row

    fig = make_subplots(
        rows=rows,
        cols=features_per_row,
        subplot_titles=list(df.index),
        horizontal_spacing=0.2,
        vertical_spacing=0.15,
    )

    for idx, (model_name, row_data) in enumerate(df.iterrows()):
        row = idx // features_per_row + 1
        col = idx % features_per_row + 1

        fig.add_trace(
            go.Bar(
                x=row_data.values,
                y=row_data.index,
                orientation="h",
                name=model_name,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        width=width_px,
        height=height_px * rows,
        title=dict(
            text=custom_title,
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.92,
            x=0.02,
            font=dict(size=22),
        ),
        showlegend=False,
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        file_name = f"plotly_charts/{image_name}.png"
        pio.write_image(fig, file_name)
        display(Image(file_name))


def plot_feature_importances_one_model(
    df: pd.DataFrame,
    model_name: str,
    width_px: int = 600,
    height_px: int = 400,
    custom_title: str = "Feature Importance",
    custom_subtitle: str = "by permuation importance mean",
    image_name: str = "feature_importances",
    plotly_mode: str = PLOTLY_MODE,
) -> None:
    """
    Visualizes feature importances for a single model using a horizontal bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where rows are models and columns are feature importances.
    model_name : str
        Name of the model whose feature importances are to be visualized.
    width_px : int, optional
        Total width of the plot in pixels.
    height_px : int, optional
        Total height of the plot in pixels.
    custom_title : str, optional
        Custom title text for the plot.
    image_name : str, optional
        Base name for saving image files when `plotly_mode` is not set to "notebook".
    plotly_mode : str, optional
        The mode in which Plotly renders the output. If "notebook", the plot
        is displayed inline.
    """
    if model_name not in df.index:
        raise ValueError(f"Model '{model_name}' not found in the DataFrame index.")

    feature_values = df.loc[model_name]
    feature_names = df.columns
    sorted_indices = feature_values.argsort()
    sorted_feature_values = feature_values.iloc[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sorted_feature_values,
            y=sorted_feature_names,
            orientation="h",
            name=model_name,
        )
    )

    fig.update_layout(
        width=width_px,
        height=height_px,
        title=dict(
            text=f"{custom_title}: {model_name}",
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="bottom",
            y=0.92,
            x=0.02,
            font=dict(size=20),
        ),
        title_subtitle=dict(text=custom_subtitle, font=dict(size=14)),
        showlegend=False,
        yaxis=dict(
            ticks="outside",
            ticklen=10,
            tickcolor="black",
        ),
        margin=dict(l=30, r=30, t=80, b=60),
    )

    if plotly_mode == "notebook":
        fig.show()
    else:
        file_name = f"plotly_charts/{image_name}_{model_name}.png"
        pio.write_image(fig, file_name)
        display(Image(file_name))


def plot_shap_summary(shap_values, transformed_data):
    """
    Plots the SHAP summary plot with dark mode styling.

    Parameters:
    - shap_values: Computed SHAP values from SHAP explainer.
    - transformed_data: Processed DataFrame with feature names.
    """

    plt.style.use("dark_background")
    shap.summary_plot(shap_values, transformed_data, show=False)
    plt.xlabel("SHAP Value", color="white")
    plt.ylabel("", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.show()
