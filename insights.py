import pandas as pd
from data_processor import *

"""Generate insights for top and bottom N entries based on a specified metric."""

def generate_insight(file, metric_col, group_col="(None)", top_n=5):
    """
    Generate insights for the top and bottom N entries based on a specified metric.

    Parameters:
    - file: UploadedFile from Gradio.
    - metric_col: The column name of the numeric metric to analyze.
    - group_col: Optional categorical column to group by. "(None)" means no grouping.
    - top_n: Number of top and bottom entries to consider.

    Returns:
    - insight_text: A markdown string containing the generated insights.
    """

    if file is None:
        return "❗ Please upload a dataset first."

    df = input_data(file)[0]  
    if df is None or df.empty:
        return "❗ The dataset is empty or could not be loaded."

    if metric_col is None:
        return "❗ Please select a metric (numeric column)."

    if metric_col not in df.columns:
        return f"❗ Selected metric column **{metric_col}** not found in the dataset."

    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        return f"❗ Column **{metric_col}** is not numeric. Please select a numeric column."

    display_col = None

    if "name" in df.columns:
        display_col = "name"
    else:
        non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric_cols:
            display_col = non_numeric_cols[0]

    if group_col is None or group_col == "(None)":
        sorted_data = df.sort_values(by=metric_col, ascending=False)

        top_entries = sorted_data.head(top_n)
        bottom_entries = sorted_data.tail(top_n)

        top_one = sorted_data.head(1)
        bottom_one = sorted_data.tail(1)

        insight_lines = []
        insight_lines.append(f"### 🔝 Top {top_n} rows by **{metric_col}**\n")

        for _, row in top_entries.iterrows():
            label = f"{display_col} = {row[display_col]}" if display_col else f"index = {row.name}"
            insight_lines.append(f"- {label}, **{metric_col}** = {row[metric_col]}")

        insight_lines.append(f"\n### 🔻 Bottom {top_n} rows by **{metric_col}**\n")
                             
        for _, row in bottom_entries.iterrows():
            label = f"{display_col} = {row[display_col]}" if display_col else f"index = {row.name}"
            insight_lines.append(f"- {label}, **{metric_col}** = {row[metric_col]}")
        
        top_label = top_one.iloc[0][display_col] if display_col else top_one.index[0]
        top_value = top_one.iloc[0][metric_col]
        
        bottom_label = bottom_one.iloc[0][display_col] if display_col else bottom_one.index[0]
        bottom_value = bottom_one.iloc[0][metric_col]

        mean_val = df[metric_col].mean()

        top_diff = ((top_value - mean_val) / mean_val) * 100
        bottom_diff = ((bottom_value - mean_val) / mean_val) * 100

        insight_lines.append(
            f"\n### 🏆 Overall Top Performer\n"
            f"- **{top_label}** leads with **{metric_col} = {top_value:.2f}**, "
            f"which is **{top_diff:.1f}% above the dataset average ({mean_val:.2f})**.\n"
        )

        insight_lines.append(
            f"### 🥉 Overall Lowest Performer\n"
            f"- **{bottom_label}** records **{metric_col} = {bottom_value:.2f}**, "
            f"which is **{abs(bottom_diff):.1f}% below the dataset average ({mean_val:.2f})**.\n"
        )

        return "\n".join(insight_lines)

    # ===== Group-by Summary Analysis =====

    if group_col not in df.columns:
        return f"❗ Group-by column **{group_col}** not found in the dataset."

    group_stats = (
        df.groupby(group_col)[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: f"{metric_col}_mean"})
    )

    group_stats_sorted = group_stats.sort_values(by=f"{metric_col}_mean", ascending=False)

    top_groups = group_stats_sorted.head(top_n)
    bottom_groups = group_stats_sorted.tail(top_n)

    overall_mean = df[metric_col].mean()

    # Build insight text
    insight_lines = []

    # ====== OVERALL SUMMARY ======
    highest_group = top_groups.iloc[0]
    lowest_group = bottom_groups.iloc[0]

    highest_diff = ((highest_group[f"{metric_col}_mean"] - overall_mean) / overall_mean) * 100
    lowest_diff = ((lowest_group[f"{metric_col}_mean"] - overall_mean) / overall_mean) * 100

    insight_lines.append("## 📈 Overall Group-level Insight Summary\n")

    insight_lines.append(
        f"- The **highest-performing group** is **`{highest_group[group_col]}`**, "
        f"with mean({metric_col}) = **{highest_group[f'{metric_col}_mean']:.2f}**, "
        f"which is **{highest_diff:.1f}% above** the overall dataset average ({overall_mean:.2f})."
    )

    insight_lines.append(
        f"- The **lowest-performing group** is **`{lowest_group[group_col]}`**, "
        f"with mean({metric_col}) = **{lowest_group[f'{metric_col}_mean']:.2f}**, "
        f"which is **{abs(lowest_diff):.1f}% below** the overall dataset average ({overall_mean:.2f})."
    )

    if highest_diff > 30:
        insight_lines.append(
            f"- This indicates a **strong cluster effect**, where `{highest_group[group_col]}` significantly outperforms all other groups."
        )
    elif highest_diff > 10:
        insight_lines.append(
            f"- This group performs **above average**, suggesting favorable characteristics for achieving higher {metric_col}."
        )
    else:
        insight_lines.append(
            f"- The top group is only slightly above average, implying **relatively even distribution** across groups."
        )

    if lowest_diff < -30:
        insight_lines.append(
            f"- The lowest-performing group has a **substantial disadvantage**, indicating structural differences."
        )
    elif lowest_diff < -10:
        insight_lines.append(
            f"- This group performs **below average**, which may signal competitive or environmental limitations."
        )
    else:
        insight_lines.append(
            f"- The bottom group is close to the dataset average, suggesting **no extreme underperformance**."
        )


    # ===== TOP Groups =====
    insight_lines.append(
        f"\n### 🔝 Top {top_n} Groups by mean({metric_col})"
    )

    for _, row in top_groups.iterrows():
        insight_lines.append(
            f"- **`{row[group_col]}`** → {row[f'{metric_col}_mean']:.2f}"
        )

    # ===== BOTTOM Groups =====
    insight_lines.append(
        f"\n### 🔻 Bottom {top_n} Groups by mean({metric_col})"
    )

    for _, row in bottom_groups.iterrows():
        insight_lines.append(
            f"- **`{row[group_col]}`** → {row[f'{metric_col}_mean']:.2f}"
        )

    return "\n".join(insight_lines)

def generate_trends_and_anomalies_from_df(file, max_numeric_cols=3, max_cat_cols=3):
    """
    Generate an automatic textual summary of basic trends and anomalies
    for the given DataFrame.

    Returns:
        markdown_str (str): Formatted markdown string.
    """

    if file is None:
        return "❗ Dataset is empty. Cannot generate insights."
    
    df = input_data(file)[0]

    n_rows = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    trend_lines = []
    anomaly_lines = []

    # =========================
    #  Overall Trend Insights
    # =========================
    trend_lines.append("### 📝 Overall Trend Insights\n")

    # ---- Numeric columns: ----
    numeric_info = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        skew = s.skew()  
        desc = s.describe()
        numeric_info.append({
            "col": col,
            "skew": skew,
            "mean": desc["mean"],
            "min": desc["min"],
            "max": desc["max"],
            "std": desc["std"],
        })

    numeric_info_sorted = sorted(numeric_info, key=lambda x: abs(x["skew"]), reverse=True)
    numeric_info_sorted = numeric_info_sorted[:max_numeric_cols]

    for info in numeric_info_sorted:
        col = info["col"]
        skew = info["skew"]

        if skew > 1:
            skew_desc = "strongly right-skewed"
        elif skew > 0.5:
            skew_desc = "moderately right-skewed"
        elif skew < -1:
            skew_desc = "strongly left-skewed"
        elif skew < -0.5:
            skew_desc = "moderately left-skewed"
        else:
            skew_desc = "roughly symmetric"

        trend_lines.append(
            f"- The distribution of `{col}` is **{skew_desc}**, "
            f"suggesting that values are not evenly spread across the range."
        )

    # ---- Categorical columns: ----
    cat_info = []
    for col in cat_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        vc = s.value_counts(normalize=True)
        top_cat = vc.index[0]
        top_pct = vc.iloc[0] * 100
        cat_info.append({
            "col": col,
            "top_cat": top_cat,
            "top_pct": top_pct,
        })

    cat_info_sorted = sorted(cat_info, key=lambda x: x["top_pct"], reverse=True)
    cat_info_sorted = cat_info_sorted[:max_cat_cols]

    for info in cat_info_sorted:
        col = info["col"]
        top_cat = info["top_cat"]
        top_pct = info["top_pct"]

        if top_pct >= 70:
            desc = "is strongly dominated"
        elif top_pct >= 50:
            desc = "is mainly composed"
        else:
            desc = "shows a noticeable preference"

        trend_lines.append(
            f"- `{col}` {desc} of **`{top_cat}` ({top_pct:.1f}%)**, "
            f"indicating an imbalanced distribution across categories."
        )

    # =========================
    #  Anomaly Detection
    # =========================
    anomaly_lines.append("\n### ⚠ Anomaly Detection\n")

    # ---- Numeric outliers (IQR rule) ----
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue  

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask_low = s < lower
        mask_high = s > upper
        n_low = int(mask_low.sum())
        n_high = int(mask_high.sum())
        n_outliers = n_low + n_high

        if n_outliers == 0:
            continue

        if n_high >= n_low:
            direction = "high-end outliers"
            count_major = n_high
        else:
            direction = "low-end outliers"
            count_major = n_low

        anomaly_lines.append(
            f"- `{col}` contains **{n_outliers} outliers** "
            f"({direction}), based on the IQR rule."
        )

    # ---- Rare categories ----
    for col in cat_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        vc = s.value_counts()
        rare = vc[vc <= max(3, 0.01 * n_rows)]
        if rare.empty:
            continue

        example_cats = ", ".join([f"`{idx}`" for idx in rare.index[:3]])
        anomaly_lines.append(
            f"- `{col}` contains **{len(rare)} rare categories** "
            f"(e.g., {example_cats}), which may have limited statistical reliability."
        )

    # ---- Missing values patterns ----
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / n_rows) * 100

    cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    for col, pct in cols_with_missing.items():
        if pct < 5:
            continue  

        if pct >= 40:
            level = "a high level"
        elif pct >= 20:
            level = "a moderate level"
        else:
            level = "a noticeable amount"

        anomaly_lines.append(
            f"- `{col}` has **{pct:.1f}% missing values**, indicating {level} of missing data."
        )

    if len(anomaly_lines) == 1: 
        anomaly_lines.append("- No strong anomalies were detected based on basic rules.")

    all_lines = trend_lines + anomaly_lines
    return "\n".join(all_lines)