import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import *

def plot_corr_matrix(df, data_categorys):
    """
    Plot correlation matrix heatmap for numerical features

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    data_categorys (dict): Dictionary containing data categories.

    Returns:
    fig (Figure): Matplotlib Figure object containing the heatmap.
    """

    corr = correlation_matrix(df, data_categorys)

    figure, ax = plt.subplots(figsize = (8, 6))
    sns.heatmap(corr, annot = False, cmap = "coolwarm", ax = ax)
    ax.set_title("Correlation Matrix Heatmap")

    return figure

def plot_time_series(file, date_col, value_col, agg_method):
    """
    Plot time series line chart for a given date column and value column

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    date_col (str): Name of the date column.
    value_col (str): Name of the value column.

    Returns:
    fig (Figure): Matplotlib Figure object containing the time series plot.
    """
    
    df = input_data(file)[0]

    if not date_col or not value_col:
        figure, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Please select both a date column and a numeric column.",
                ha="center", va="center")
        ax.axis("off")
        return figure
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])

    # Convert datetime → date only (daily aggregation)
    df["__date__"] = df[date_col].dt.date

    # Perform aggregation
    if agg_method in ["mean", "sum", "median", "count"]:
        grouped = (
            df.groupby("__date__")[value_col]
            .agg(agg_method)
            .reset_index()
        )
    else:
        grouped = df[["__date__", value_col]]

    # Plot
    figure, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data = grouped, x = "__date__", y = value_col, ax = ax)

    ax.set_title(f"{agg_method.capitalize()} of {value_col} over time")
    ax.set_xlabel(date_col)
    ax.set_ylabel(f"{agg_method}({value_col})")

    plt.xticks(rotation = 45)

    return figure

def plot_distribution(file, numeric_col, plot_type):
    """
    Plot distribution of a numeric column using histogram or box plot

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    col (str): Name of the numeric column.
    plot_type (str): Type of plot ("Histogram" or "Box Plot").

    Returns:
    fig (Figure): Matplotlib Figure object containing the distribution plot.
    """

    if file is None or not numeric_col:
        return None

    df = input_data(file)[0]
    col_data = df[numeric_col].dropna()

    low, high = np.percentile(col_data, [1, 99])
    col_data_clipped = col_data[(col_data >= low) & (col_data <= high)]

    figure, ax = plt.subplots(figsize=(8, 5))

    if plot_type == "Histogram":
        ax.hist(col_data_clipped, bins=30)
        ax.set_title(f"Histogram of {numeric_col}")
        ax.set_xlabel(numeric_col)
        ax.set_ylabel("Count")

    elif plot_type == "Box Plot":
        ax.boxplot(col_data_clipped, vert=False, showfliers=True)
        ax.set_title(f"Box Plot of {numeric_col}")
        ax.set_xlabel(numeric_col)

    return figure

def plot_categorical_distribution(file, categorical_col, chart_type):
    """
    Plot bar chart of value counts for a categorical column

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    col (str): Name of the categorical column.

    Returns:
    fig (Figure): Matplotlib Figure object containing the bar chart.
    """

    if file is None or not categorical_col:
        return None

    df = input_data(file)[0]
    counts = df[categorical_col].value_counts()

    figure, ax = plt.subplots(figsize=(8, 6))

    if chart_type == "Bar Chart":
        sns.barplot(x = counts.index, y = counts.values, ax=ax)
        ax.set_title(f"Bar Chart of {categorical_col}")
        ax.set_xlabel(categorical_col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)

    elif chart_type == "Pie Chart":
        ax.pie(counts.values, labels = counts.index, autopct = "%1.1f%%")
        ax.set_title(f"Pie Chart of {categorical_col}")

    return figure

def plot_scatter(file, x_col, y_col):
    """
    Plot scatter plot between two numeric columns.

    Args:
        file: Uploaded file object
        x_col (str): name of column for x-axis
        y_col (str): name of column for y-axis

    Returns:
        fig (Figure): Matplotlib Figure object
    """

    if file is None or x_col is None or y_col is None:
        return None

    df = input_data(file)[0]

    if x_col not in df.columns or y_col not in df.columns:
        return None

    figure, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

    ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    return figure

def export_time_series_plot_png(file, date_col, numeric_col, agg_method):
    """
    Export time series plot as PNG file

    Args:
    file: Uploaded file object
    date_col (str): Name of the date column.
    numeric_col (str): Name of the numeric column.
    agg_method (str): Aggregation method for the numeric column.

    Returns:
    file_path (str): Path to the saved PNG file.
    """

    if file is None or not date_col or not numeric_col:
        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5,
            "No data / column selected",
            ha="center", va="center"
        )
        ax.axis("off")
    else:
        fig = plot_time_series(file, date_col, numeric_col, agg_method)

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, "time_series_plot.png")

    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path


def export_distribution_plot_png(file, numeric_col, plot_type):
    """
    Export distribution plot as a PNG file.

    Args:
    file (str): Path to the data file.
    dist_col (str): Column name for distribution plot.
    dist_type (str): Type of distribution plot (e.g., histogram, boxplot).

    Returns:
    file_path (str): Path to the saved PNG file.
    """

    if file is None or not numeric_col:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data / column selected",
                ha="center", va="center")
        ax.axis("off")
    else:
        fig = plot_distribution(file, numeric_col, plot_type)

        if fig is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data / column selected",
                    ha="center", va="center")
            ax.axis("off")

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, "distribution_plot.png")

    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path

def export_categorical_plot_png(file, categorical_col, chart_type):
    """
    Export categorical plot (bar / pie) as PNG file.

    Args:
    file (str): Path to the data file.
    categorical_col (str): Column name for categorical plot.
    chart_type (str): Type of categorical plot (e.g., bar chart, pie chart

    Returns:
    file_path (str): Path to the saved PNG file.
    """

    if file is None or not categorical_col:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data / column selected",
                ha="center", va="center")
        ax.axis("off")
    else:
        fig = plot_categorical_distribution(file, categorical_col, chart_type)
        if fig is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data / column selected",
                    ha="center", va="center")
            ax.axis("off")

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, "categorical_plot.png")

    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path

def export_scatter_plot_png(file, x_col, y_col):
    """
    Export scatter plot as PNG file.

    Args:
    file (str): Path to the data file.
    x_col (str): Column name for x-axis.
    y_col (str): Column name for y-axis.

    Returns:
    file_path (str): Path to the saved PNG file.
    """

    if file is None or not x_col or not y_col:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data / column selected",
                ha="center", va="center")
        ax.axis("off")
    else:
        fig = plot_scatter(file, x_col, y_col)
        if fig is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data / column selected",
                    ha="center", va="center")
            ax.axis("off")

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, "scatter_plot.png")

    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    return file_path
