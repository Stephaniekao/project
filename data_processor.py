import pandas as pd
import numpy as np
import tempfile
import os

"""
Data loading, cleaning, filtering

1. Data Upload & Validation
    - Upload CSV or Excel files through Gradio interface
    - Display basic dataset information (shape, columns, data types)
    - Show data preview (first/last N rows)
    - Handle common data issues gracefully with informative error messages
"""

def input_data(file):
    """
    Upload CSV or Excel files and returns a pandas DataFrame.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    df (DataFrame): Pandas DataFrame containing the data.
    """

    # Check if file is provided
    if file is None:
        return "No file uploaded."
    
    # Try to read the file based on its extension
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name, encoding='latin1')
        
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file.name, encoding='latin1')
        
        else:
            return "Unsupported file format. Please upload a CSV or Excel file."
    
    # Handle potential errors during file reading
    except Exception as e:
        return f"Error loading file: {e}"
    
    return df, "✅ File uploaded successfully!"

def data_info(df):
    """
    Display basic dataset information (shape, columns, data types)

    Args:
    df (DataFrame): Pandas DataFrame containing the data.

    Returns:
    df_info (dict): Dictionary containing dataset information.
    """

    df_info = {} # Initialize an empty dictionary to hold the info
    df_info['Shape'] = df.shape # (rows, columns)
    df_info['Columns'] = df.columns.tolist() # List of column names
    df_info['Data Types'] = df.dtypes.astype(str).to_dict() # Data types of each column
    
    return df_info

def data_preview(df, n=5):
    """
    Show data preview (first/last N rows)

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    n (int): Number of rows to display from the start and end.

    Returns:
    preview (dict): Dictionary containing head and tail previews.
    """

    head = df.head(n)
    tail = df.tail(n)

    sep = pd.DataFrame({col: ["..."] for col in df.columns})

    preview = pd.concat([head, sep, tail], ignore_index=True)

    return preview

def data_info_error(df):
    """
    Validate dataset and return error message if the dataset is unusable.
    Return None if everything is OK.
    """

    # 1. No file loaded
    if df is None:
        return "No data loaded. Please upload a CSV or Excel file."

    # 2. Wrong type
    if not isinstance(df, pd.DataFrame):
        return "Invalid data format. Please upload a CSV or Excel file."

    # 3. Empty file
    if df.empty:
        return "The uploaded file is empty. Please upload a valid dataset."

    # 4. Entire dataset is NaN
    if df.isna().all().all():
        return "The dataset contains only missing values and cannot be analyzed."
    
    # 5. Column entirely empty (all NaN)
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        return f"The following columns contain no data: {', '.join(empty_cols)}."

    # 6. Columns with only 1 unique value (no useful information)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        return f"The following columns have no variation and may not be useful: {', '.join(constant_cols)}."

    # If none of the above triggered → dataset is OK
    return None

def detect_datetime_columns(df, threshold=0.9):
    """
    Detect datetime columns in the DataFrame.

    Args:
    df (DataFrame): Pandas DataFrame containing the data.

    Returns:
    datetime_cols (list): List of column names detected as datetime.
    """
    datetime_cols = []

    for col in df.columns:
        series = df[col].astype(str)

        # Fast check：欄位是否含有 date-like pattern
        if not series.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).any():
            continue

        # Try convert
        converted = pd.to_datetime(series, errors="coerce")

        # If at least threshold% rows look like dates → treat as datetime
        if converted.notna().mean() >= threshold:
            datetime_cols.append(col)

    return datetime_cols

def data_category(df):
    """
    Display categorical columns and their unique values
    """
    datetime_cols = detect_datetime_columns(df, threshold=0.9)
    if datetime_cols:
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, errors="coerce")

    data_categorys = {}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    data_categorys["Numeric Columns"] = numeric_cols
    data_categorys["Categorical Columns"] = categorical_cols
    data_categorys["Datetime Columns"] = datetime_cols

    return data_categorys

def data_numerical_metrics(df, data_categorys):
    """
    Numerical columns: mean, median, std, min, max, quartiles

    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    data_categorys (dict): Dictionary containing data categories.

    Returns:
    numerical_stats (dict): Dictionary containing numerical statistics.
    """

    # Initialize dictionaries to hold statistics
    numerical_stats = {}

    # Get numerical columns mean, median, std, min, max, quartiles
    numerical_cols = data_categorys.get("Numeric Columns", [])

    # Calculate statistics for each numerical column
    for col in numerical_cols:
        numerical_stats[col] = {}

        numerical_mean = df[col].mean()
        numerical_stats[col]["mean"] = float(df[col].mean())
    
        numerical_median = df[col].median()
        numerical_stats[col]['median'] = float(df[col].median())
    
        numerical_std = df[col].std()
        numerical_stats[col]['std'] = float(df[col].std())
    
        numerical_min = df[col].min()
        numerical_stats[col]['min'] = float(df[col].min())
        
        numerical_max = df[col].max()
        numerical_stats[col]['max'] = float(df[col].max())
    
        numerical_quartiles = df[col].quantile([0.25, 0.5, 0.75])
        numerical_stats[col]['quartiles'] = {
            "25%": float(numerical_quartiles.loc[0.25]),
            "50%": float(numerical_quartiles.loc[0.5]),
            "75%": float(numerical_quartiles.loc[0.75]),
        }
        numerical_stats[col] = numerical_stats[col]
        
    return numerical_stats

def data_categorical_metrics(df, data_categorys):
    """
    Categorical columns: unique values, value counts, mode
    
    Args:
    df (DataFrame): Pandas DataFrame containing the data.
    data_categorys (dict): Dictionary containing data categories.

    Returns:
    category_stats (dict): Dictionary containing categorical statistics.
    """

    # Initialize dictionaries to hold statistics
    category_stats = {}

    # Get categorical columns
    category_cols = data_categorys.get("Categorical Columns", [])

    for col in category_cols:

        category_stats[col] = {}

        # Unique values (excluding NaN)
        n_unique = int(df[col].nunique(dropna=True))

        # Value counts (include NaN as a key if present)
        value_counts = df[col].value_counts(dropna=False)

        # Top-k frequent values (as dict)
        top_k = value_counts.head(10)

        # Normalized frequencies (proportion of total rows)
        value_counts_norm = (value_counts / len(df)).round(4)

        # Mode: may be multiple values; convert index to Python list
        max_count = value_counts.max()
        mode_vals = value_counts[value_counts == max_count].index.tolist()

        # Store statistics in the dictionary
        category_stats[col] = {
            'n_unique': n_unique,
            'value_counts': value_counts.to_dict(),
            'top_counts': top_k.to_dict(),
            'value_counts_normalized': value_counts_norm.to_dict(),
            'mode': [None if pd.isna(v) else v for v in mode_vals]
        }

    return category_stats

def auto_data_check(df):
    """
    Correlation matrix for numerical features
    Generate automated data profiling report using pandas-profiling

    Args:
    df (DataFrame): Pandas DataFrame containing the data.

    Returns:
    profile (ProfileReport): Pandas Profiling Report object.
    """
    data_check = {}

    # Missing values analysis
    missing_counts = df.isna().sum()
    data_check['missing_counts'] = missing_counts

    # Missing values percentage analysis
    missing_percentage = (df.isna().mean() * 100).round(2)
    data_check['missing_percentage'] = missing_percentage

    # Duplicate rows analysis
    duplicate_count = df.duplicated().sum()
    data_check['duplicate_count'] = duplicate_count

    # Duplicate rows percentage analysis
    duplicate_percentage = round(df.duplicated().mean() * 100, 2)
    data_check['duplicate_percentage'] = duplicate_percentage

    return data_check

def correlation_matrix(df, data_categorys):
    """
    Compute correlation matrix for numerical features

    Args:
    df (DataFrame): Pandas DataFrame containing the data.

    Returns:
    corr_matrix (DataFrame): Correlation matrix DataFrame.
    """

    # Get numerical columns
    numeric_cols = data_categorys.get("Numeric Columns", [])

    # Compute correlation matrix for numerical columns
    corr_matrix = df[numeric_cols].corr()

    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    return corr_matrix

def export_filtered_csv(filtered_df):
    """
    Export the full filtered DataFrame as a downloadable CSV file.

    Args:
    filtered_df (DataFrame): Pandas DataFrame containing the filtered data.

    Returns:
    file_path (str): Path to the saved CSV file.
    """

    if filtered_df is None or len(filtered_df) == 0:
        csv_str = "message\nNo filtered data available."
    else:
        csv_str = filtered_df.to_csv(index=False)

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, "filtered_data.csv")

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_str)

    return file_path
