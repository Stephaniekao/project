"""
Main utility functions for Gradio app.
"""

import gradio as gr
from data_processor import *
from visualizations import *
from insights import *

class ParisAirbnbDashboard:
    """
    Gradio app for Paris Airbnb Business Intelligence Dashboard
    """
    def __init__(self):
        """
        Initialize the Gradio interface components.
        """

        self.file_input = None
        self.process_button = None
        self.demo = None
        self.data_state = None  
        self.status = None

    def upload_file(self):
        """
        Gradio interface to upload CSV or Excel files and display basic dataset information.
        """
        
        # File upload component
        self.file_input = gr.File(label="Upload your CSV or Excel file here", file_types=['.csv', '.xlsx'])  

        # Button to trigger data processing
        self.process_button = gr.Button("Upload Data")   

        # State to hold uploaded data
        self.data_state = gr.State()

        # Display a success message UI
        self.status = gr.Markdown("")

        # Define button click action
        self.process_button.click(fn = input_data, inputs = self.file_input, outputs = [self.data_state, self.status]) 

        return self.file_input, self.process_button

    def format_columns_multicol(self, columns, cols_per_row=5):
        """
        Format column names into multiple columns for better readability in Gradio.

        Args:
        columns (list): List of column names.
        cols_per_row (int): Number of columns per row.

        Returns:
        formatted_str (str): Formatted string with multiple columns.
        """

        rows = []
        for i in range(0, len(columns), cols_per_row):
            group = columns[i:i+cols_per_row]
            rows.append("{:<30} {:<30} {:<30} {:<30} {:<30}".format(*group + [""]*(cols_per_row - len(group))))

        return "```\n" + "\n".join(rows) + "\n```"
    
    def format_types_multicol(self, type_dict, col_width=30):
        """ 
        Format data types into a multi-column string for better readability in Gradio.

        Args:
        type_dict (dict): Dictionary with column names as keys and data types as values.
        col_width (int): Width of each column for alignment.

        Returns:
        formatted_str (str): Formatted string with multiple columns.
        """

        rows = []
        for col, dtype in type_dict.items():
            rows.append(f"{col:<{col_width}} {dtype}")

        return "```\n" + "\n".join(rows) + "\n```"

    def data_info_markdown(self, df_info):
        """
        Convert DataFrame info into a markdown string for Gradio display.

        Args:
        df_info (str): String output from DataFrame.info().

        Returns:
        markdown_str (str): Formatted markdown string.
        """
        md_info = []

        # Shape
        rows, cols = df_info["Shape"]
        md_info.append("\n### 📐 Shape")
        md_info.append("<details><summary>Click to expand</summary>\n")
        md_info.append(f"- **Rows**: {rows}")
        md_info.append(f"- **Columns**: {cols}")
        md_info.append("\n</details>")

        # Columns list
        md_info.append("\n### 🏷️ Column Names")
        md_info.append("<details><summary>Click to expand</summary>\n")
        md_info.append(self.format_columns_multicol(df_info["Columns"]))
        md_info.append("\n</details>")

        # Data types
        md_info.append("\n### 🔢 Data Types")
        md_info.append("<details><summary>Click to expand</summary>\n")
        for col, dtype in df_info["Data Types"].items():
            md_info.append(f"- **{col}**: `{dtype}`")

        md_info.append("\n</details>")
        return "\n".join(md_info)

    def show_info(self, file):
        """
        Display basic dataset information in markdown format.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        str: Markdown formatted string with dataset information.
        """
        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]
        
        # Get dataset info and convert to markdown
        info = data_info(df)
        info_md = self.data_info_markdown(info)

        return info_md
    
    def show_preview(self, file, n=5):
        """
        Display basic dataset information in markdown format.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        str: Markdown formatted string with dataset information.
        """
        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]
        
        # Get dataset preview and convert to markdown
        preview_df = data_preview(df, n)

        return ("", gr.update(value=preview_df, visible=True))
        
    def validate_data(self, file):
        """
        Validate dataset and return error message if the dataset is unusable.

        Args:
        df (pd.DataFrame): The DataFrame to validate.

        Returns:
        str: Error message if any issues found, otherwise success message.
        """
        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]

        # Validate dataset and get error message if any
        error_msg = data_info_error(df)

        if error_msg is None:
            return "✅ No issues found. The dataset looks good!"
        else:
            return f"⚠️ {error_msg}"

    def compute_selected_metrics(self, df, numeric_selected, categorical_selected):
        """
        Compute selected metrics for numerical and categorical columns

        Args:
        df (DataFrame): Pandas DataFrame containing the data.
        numeric_selected (list): List of selected numerical columns.
        categorical_selected (list): List of selected categorical columns.

        Returns:
        selected_metrics (dict): Dictionary containing selected metrics.
        """

        data_catagorys = data_category(df)

        if numeric_selected:
            data_catagorys["Numeric Columns"] = numeric_selected
        else:
            data_catagorys["Numeric Columns"] = []

        if categorical_selected:
            data_catagorys["Categorical Columns"] = categorical_selected    
        else:
            data_catagorys["Categorical Columns"] = []

        numerical_stats = data_numerical_metrics(df, data_catagorys)
        categorical_stats = data_categorical_metrics(df, data_catagorys)

        return {
            "Numerical Metrics": numerical_stats,
            "Categorical Metrics": categorical_stats,
        }

    def update_metrics_choosers(self, file):
        """
        Update the column selectors for metrics exploration based on uploaded dataset.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        tuple: Lists of numerical and categorical and datetime column names.
        """
        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]
        data_categorys = data_category(df)

        numerical_cols = data_categorys["Numeric Columns"]
        categorical_cols = data_categorys["Categorical Columns"]

        return (
            gr.update(choices = numerical_cols, value=[]),
            gr.update(choices = categorical_cols, value=[]),
        )

    def numerical_md_table(self, numerical_stats):
        """
        Convert numerical statistics to a markdown table.

        Args:
        numerical_stats (dict): Dictionary containing numerical statistics.

        Returns:
        str: Markdown formatted table.
        """
        md_metrix = []

        for col, stats in numerical_stats.items():
            md_metrix.append(f"### 📈 {col}\n")
            md_metrix.append("| Metric | Value |")
            md_metrix.append("|---|---|")
            for stat_name, stat_value in stats.items():
                md_metrix.append(f"| {stat_name} | {stat_value} |")
            md_metrix.append("\n")

        return "\n".join(md_metrix)

    def categorical_md_table(self, categorical_stats):
        """
        Convert categorical statistics to a markdown table.

        Args:
        categorical_stats (dict): Dictionary containing categorical statistics.

        Returns:
        str: Markdown formatted table.
        """

        md_metrix = []

        for col, stats in categorical_stats.items():
        
            md_metrix.append(f"### 📊 {col}\n")

            # n_unique
            md_metrix.append(f"- **n_unique:** {stats['n_unique']}")

            # 🔝 top_counts
            md_metrix.append(
                "<details><summary><strong>🔝 top_counts</strong> (Top 10)</summary>\n\n"
            )
            md_metrix.append("| Category | Count |")
            md_metrix.append("|----------|-------|")

            for key, val in stats["top_counts"].items():
                display_key = "None" if pd.isna(key) else str(key)
                md_metrix.append(f"| {display_key} | {val} |")

            md_metrix.append("\n</details>\n")

            # 📦 value_counts
            md_metrix.append(
                "<details><summary><strong>📦 value_counts</strong></summary>\n\n"
            )
            md_metrix.append("| Category | Count |")
            md_metrix.append("|----------|-------|")

            for key, val in stats["value_counts"].items():
                display_key = "None" if pd.isna(key) else str(key)
                md_metrix.append(f"| {display_key} | {val} |")

            md_metrix.append("\n</details>\n")

            # 📊 value_counts_normalized
            md_metrix.append(
                "<details><summary><strong>📊 value_counts_normalized</strong></summary>\n\n"
            )
            md_metrix.append("| Category | Frequency |")
            md_metrix.append("|----------|-----------|")

            for key, val in stats["value_counts_normalized"].items():
                display_key = "None" if pd.isna(key) else str(key)
                md_metrix.append(f"| {display_key} | {val} |")

            md_metrix.append("\n</details>\n")

            md_metrix.append(f"- **mode:** {stats['mode']}\n")

        return "\n".join(md_metrix)
    
    def show_selected_metrics(self, file, numeric_selected, categorical_selected):
        """
        Show selected metrics for chosen columns.

        Args:
        file: Uploaded file object from Gradio.
        numeric_selected (list): List of selected numerical columns.
        categorical_selected (list): List of selected categorical columns.

        Returns:
        str: Markdown formatted metrics for selected columns.
        """

        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]

        selected_metrics = self.compute_selected_metrics(
            df,
            numeric_selected or [],
            categorical_selected or []
        )

        numerical_stats = selected_metrics["Numerical Metrics"]
        categorical_stats = selected_metrics["Categorical Metrics"]

        num_md = self.numerical_md_table(numerical_stats) if numerical_stats else "⚠️ No numerical columns selected.\n"
        cat_md = self.categorical_md_table(categorical_stats) if categorical_stats else "⚠️ No categorical columns selected.\n"

        return num_md + "\n\n" + cat_md

    def data_auto_markdown(self, df_info):
        """
        Convert DataFrame info into a markdown string for Gradio display.

        Args:
        df_info (str): String output from DataFrame.info().

        Returns:
        markdown_str (str): Formatted markdown string.
        """
        md_auto = []

        missing_counts = df_info.get("missing_counts")
        missing_percentage = df_info.get("missing_percentage")
        duplicate_count = df_info.get("duplicate_count")
        duplicate_percentage = df_info.get("duplicate_percentage")

        md_auto.append("\n### 🔁 Missing Values Overview")
        md_auto.append("<details><summary>Click to expand</summary>\n")
        md_auto.append("| Column | Missing Count | Missing % |")
        md_auto.append("|---|---:|---:|")

        for col in missing_counts.index:
            count = int(missing_counts[col])
            pct = float(missing_percentage[col])
            md_auto.append(f"| {col} | {count} | {pct:.2f}% |")

        md_auto.append("</details>\n")

        md_auto.append("\n### ❗ Duplicate Rows Summary\n")
        md_auto.append(f"- **Number of duplicate rows**: `{duplicate_count}`")
        md_auto.append(f"- **Percentage of duplicate rows**: `{duplicate_percentage}%`")

        return "\n".join(md_auto)

    def show_auto(self, file):
        """
        Display basic dataset information in markdown format.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        str: Markdown formatted string with dataset information.
        """
        if file is None:
            return "⚠️ Please upload a file."

        df = input_data(file)[0]
        
        # Get dataset info and convert to markdown
        auto = auto_data_check(df)
        auto_md = self.data_auto_markdown(auto)

        return auto_md

    def update_filter_column_choices(self, file, category_choice):
        """
        Update the column selector for filtering based on selected category.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.
        category_choice (str): Selected data category.

        Returns:
        list: List of column names in the selected category.
        """
        if file is None:
            return gr.update(choices=[], value=None)

        df = input_data(file)[0]
        data_categories = data_category(df)

        if category_choice == "Numeric Columns":
            cols = data_categories.get("Numeric Columns", [])
        elif category_choice == "Categorical Columns":
            cols = data_categories.get("Categorical Columns", [])
        elif category_choice == "Datetime Columns":
            cols = data_categories.get("Datetime Columns", [])
        else:
            cols = []

        return gr.update(choices = cols, value = None)

    def on_filter_category_change(self, file, category_choice):
        """
        Handle changes in filter category selection.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.
        category_choice (str): Selected data category.

        Returns:
        tuple: Updated components visibility and choices.
        """

        if file is None or not category_choice:
            return (
                # 1. filter_column
                gr.update(choices=[], value=None, visible=False),
                # 2. numeric_status
                gr.update(
                    value="*(Select a numeric column to see its range.)*",
                    visible=False
                ),
                # 3. min_slider
                gr.update(visible=False),
                # 4. max_slider
                gr.update(visible=False),
                # 5. apply_numeric_btn
                gr.update(visible=False),
                # 6. filter_result_md
                gr.update(visible=False),
                # 7. cat_status
                gr.update(
                    value="*(Select a categorical column to see its categories.)*",
                    visible=False
                ),
                # 8. cat_values
                gr.update(choices=[], value=[], visible=False),
                # 9. apply_cat_btn
                gr.update(visible=False),
                #10. filter_result_df
                gr.update(visible=False),
                #11. date_status
                gr.update(
                    value="*(Select a datetime column to see its date range.)*",
                    visible=False
                ),
                #12. start_date
                gr.update(visible=False),
                #13. end_date
                gr.update(visible=False),
                #14. apply_date_btn
                gr.update(visible=False),
            )

        df = input_data(file)[0]
        data_categories = data_category(df)

        # ---------- Numeric ----------
        if category_choice == "Numeric Columns":
            cols = data_categories.get("Numeric Columns", [])
            return (
                gr.update(choices=cols, value=None, visible=True),   # filter_column
                gr.update(
                    value="*(Select a numeric column to see its range.)*",
                    visible=True
                ),  # numeric_status
                gr.update(visible=True),   # min_slider
                gr.update(visible=True),   # max_slider
                gr.update(visible=True),   # apply_numeric_btn
                gr.update(visible=False),  # filter_result_md
                gr.update(visible=False),  # cat_status
                gr.update(choices=[], value=[], visible=False),   # cat_values
                gr.update(visible=False),  # apply_cat_btn
                gr.update(visible=False),  # filter_result_df
                gr.update(visible=False),  # date_status
                gr.update(visible=False),  # start_date
                gr.update(visible=False),  # end_date
                gr.update(visible=False),  # apply_date_btn
            )

        # ---------- Categorical ----------
        if category_choice == "Categorical Columns":
            cols = data_categories.get("Categorical Columns", [])
            return (
                gr.update(choices=cols, value=None, visible=True),   # filter_column
                gr.update(visible=False),  # numeric_status
                gr.update(visible=False),  # min_slider
                gr.update(visible=False),  # max_slider
                gr.update(visible=False),  # apply_numeric_btn
                gr.update(visible=False),  # filter_result_md
                gr.update(
                    value="*(Select a categorical column to see its categories.)*",
                    visible=True
                ),  # cat_status
                gr.update(choices=[], value=[], visible=True),   # cat_values
                gr.update(visible=True),   # apply_cat_btn
                gr.update(visible=False),  # filter_result_df
                gr.update(visible=False),  # date_status
                gr.update(visible=False),  # start_date
                gr.update(visible=False),  # end_date
                gr.update(visible=False),  # apply_date_btn
            )

        # ---------- Datetime ----------
        if category_choice == "Datetime Columns":
            cols = data_categories.get("Datetime Columns", [])
            return (
                gr.update(choices=cols, value=None, visible=True),   # filter_column
                gr.update(visible=False),  # numeric_status
                gr.update(visible=False),  # min_slider
                gr.update(visible=False),  # max_slider
                gr.update(visible=False),  # apply_numeric_btn
                gr.update(visible=False),  # filter_result_md
                gr.update(visible=False),  # cat_status
                gr.update(choices=[], value=[], visible=False),   # cat_values
                gr.update(visible=False),  # apply_cat_btn
                gr.update(visible=False),  # filter_result_df
                gr.update(
                    value="*(Select a datetime column to see its date range.)*",
                    visible=True
                ),  # date_status
                gr.update(visible=True),   # start_date
                gr.update(visible=True),   # end_date
                gr.update(visible=True),   # apply_date_btn
            )

    def update_numeric_range_status(self, file, category_choice, selected_col):
        """
        Update the numeric range status based on selected column.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.
        selected_col (str): Selected numerical column.

        Returns:
        str: Markdown formatted string with numeric range information.
        """

        if file is None or category_choice != "Numeric Columns":
            return (
                "*(Select a numeric column to see its range.)*", 
                gr.update(visible=False), 
                gr.update(visible=False),
            )

        df = input_data(file)[0]

        if not selected_col or selected_col not in df.columns:
            return "⚠️ Please select a numeric column.", gr.update(visible = False), gr.update(visible = False)

        col_data = df[selected_col].dropna()

        if col_data.empty:
            return "⚠️ Selected column has no valid numerical data.", gr.update(visible=False), gr.update(visible=False)

        if not pd.api.types.is_numeric_dtype(col_data):
            return (
                "⚠️ Selected column is not numeric. Please choose a numeric column.",
                gr.update(visible=False),
                gr.update(visible=False),
            )

        col_min = float(col_data.min())
        col_max = float(col_data.max())

        status_md = (
            f"### 🔢 Numerical Filter Status\n\n"
            f"- Selected column: **{selected_col}**\n"
            f"- Full data range: `{col_min}` ~ `{col_max}`\n"
            f"- Use the slider below to choose a filter range."
        )

        min_update = gr.update(minimum=col_min, maximum=col_max, value=col_min, visible=True)
        max_update = gr.update(minimum=col_min, maximum=col_max, value=col_max, visible=True)

        return status_md, min_update, max_update
    
    def update_categorical_values(self, file, category_choice, selected_col):
        """
        Update the category options for a selected categorical column.

        Args:
            file: Uploaded file object
            category_choice (str): Selected category type ("Categorical Columns", etc.)
            selected_col (str): Selected categorical column

        Returns:
            tuple: (Checkbox choices update, status markdown)
        """

        if file is None or category_choice != "Categorical Columns":
            return (
                gr.update(choices=[], value=[], visible=False),  # cat_values
                gr.update(                                      # cat_status
                    value="*(Select a categorical column to see its categories.)*",
                    visible=False
                ),
            )

        df = input_data(file)[0]

        if not selected_col or selected_col not in df.columns:
            return (
                gr.update(choices=[], value=[], visible=False),
                "⚠️ Please select a categorical column.",
            )

        col_data = df[selected_col].dropna().astype(str)
        unique_vals = sorted(col_data.unique().tolist())

        if not unique_vals:
            return (
                gr.update(choices=[], value=[], visible=False),
                "⚠️ This column has no valid categories.",
            )

        status_md = (
            f"### 🧩 Categorical Filter Status\n\n"
            f"- Column: **{selected_col}**\n"
            f"- Unique categories found: **{len(unique_vals)}**\n"
            f"- Select one or more categories below to filter."
        )

        return (
            gr.update(choices=unique_vals, value=[], visible=True),  # cat_values
            status_md,                                              # cat_status
        )

    def update_datetime_range_status(self, file, category_choice, selected_col):
        """
        Update the datetime range status based on selected column.

        Args:
        df (pd.DataFrame): The DataFrame to analyze.
        selected_col (str): Selected datetime column.
        
        Returns:
        str: Markdown formatted string with datetime range information.
        """

        if file is None or category_choice != "Datetime Columns":
            return (
                "*(Select a datetime column to see its date range.)*",
                gr.update(choices=[], value=None, visible=False),
                gr.update(choices=[], value=None, visible=False),
            )   

        df = input_data(file)[0]    

        if not selected_col or selected_col not in df.columns:
            return (
                "⚠️ Please select a datetime column.",
                gr.update(choices=[], value=None, visible=False),
                gr.update(choices=[], value=None, visible=False),
            )   

        col_dt = pd.to_datetime(df[selected_col], errors="coerce").dropna() 

        if col_dt.empty:
            return (
                "⚠️ Selected column has no valid datetime data.",
                gr.update(choices=[], value=None, visible=False),
                gr.update(choices=[], value=None, visible=False),
            )   

        unique_dates = sorted(col_dt.dt.date.unique())
        date_choices = [d.isoformat() for d in unique_dates]    

        status_md = (
            f"### ⏰ Datetime Filter Status\n\n"
            f"- Selected column: **{selected_col}**\n"
            f"- Full date range: `{unique_dates[0]}` ~ `{unique_dates[-1]}`\n"
            f"- Pick start and end dates from the dropdowns below."
        )   

        return (
            status_md,
            gr.update(choices=date_choices, value=date_choices[0], visible=True),   # start_date
            gr.update(choices=date_choices, value=date_choices[-1], visible=True),  # end_date
        )
    
    def apply_numeric_filter(self, file, selected_col, min_val, max_val):
        """
        Apply numeric filter to the dataset based on selected column and range.

        Args:
        df (pd.DataFrame): The DataFrame to filter.
        selected_col (str): Selected numerical column.
        range_values (tuple): Tuple containing (min_val, max_val).

        Returns:
        tuple: Markdown status and filtered DataFrame preview.
        """

        if file is None:
            return "⚠️ Please upload a file.", gr.update(visible=False)
    
        df = input_data(file)[0]
    
        if not selected_col or selected_col not in df.columns:
            return "⚠️ Please select a numeric column.", gr.update(visible=False)

        if min_val is None or max_val is None:
            return "⚠️ Please select a valid range.", gr.update(visible=False)

        low = min(min_val, max_val)
        high = max(min_val, max_val)
    
        filtered_df = df[(df[selected_col] >= low) & (df[selected_col] <= high)]
    
        status_md = (
            f"### 🔢 Numerical Filter Applied\n\n"
            f"- Column: **{selected_col}**\n"
            f"- Range: `{low}` to `{high}`\n"
            f"- Rows after filter: {len(filtered_df)}"
            f"- Rows after filter: **{len(filtered_df)} / {len(df)}**"
        )
    
        return (
            gr.update(value = status_md, visible = True),                   # filter_result_md
            gr.update(value = filtered_df.head(10), visible = True),        # filter_result_df
            filtered_df,                                       
        )

    def apply_categorical_filter(self, file, selected_col, selected_values):
        """
        Apply categorical filter based on selected values.  

        Args:
            file: Uploaded file object
            selected_col (str): Selected categorical column
            selected_values (list): Selected category values    

        Returns:
            tuple: (status markdown, filtered df preview)
        """

        if file is None:
            return "⚠️ Please upload a file.", gr.update(visible=False) 

        df = input_data(file)[0]    

        if not selected_col or selected_col not in df.columns:
            return "⚠️ Please select a categorical column.", gr.update(visible=False)   

        if not selected_values:
            return "⚠️ Please select at least one category.", gr.update(visible=False)  

        filtered_df = df[df[selected_col].astype(str).isin(selected_values)]    

        status_md = (
            f"### 🧩 Categorical Filter Applied\n\n"
            f"- Column: **{selected_col}**\n"
            f"- Selected categories: `{selected_values}`\n"
            f"- Rows after filter: {len(filtered_df)}"
            f"- Rows after filter: **{len(filtered_df)} / {len(df)}**"
        )   

        return (
            gr.update(value = status_md, visible = True),                   # filter_result_md
            gr.update(value = filtered_df.head(10), visible = True),        # filter_result_df
            filtered_df,                                       
        )

    def apply_datetime_filter(self, file, selected_col, start_date, end_date):
        """
        Apply datetime filter based on selected date range.

        Args:
        df (pd.DataFrame): The DataFrame to filter.
        selected_col (str): Selected datetime column.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
        tuple: Markdown status and filtered DataFrame preview.
        """

        if file is None:
            return "⚠️ Please upload a file.", gr.update(visible=False)

        df = input_data(file)[0]

        if not selected_col or selected_col not in df.columns:
            return "⚠️ Please select a datetime column.", gr.update(visible=False)

        col_dt = pd.to_datetime(df[selected_col], errors="coerce")

        if col_dt.isna().all():
            return "⚠️ Selected column has no valid datetime values.", gr.update(visible=False)

        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = col_dt.min()

        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = col_dt.max()

        if start > end:
            start, end = end, start

        mask = (col_dt >= start) & (col_dt <= end)
        filtered_df = df[mask]

        status_md = (
            f"### ⏰ Datetime Filter Applied\n\n"
            f"- Column: **{selected_col}**\n"
            f"- Range: `{start.date()}` to `{end.date()}`\n"
            f"- Rows after filter: {len(filtered_df)}"
            f"- Rows after filter: **{len(filtered_df)} / {len(df)}**"
        )

        return (
            gr.update(value = status_md, visible = True),                   # filter_result_md
            gr.update(value = filtered_df.head(10), visible = True),        # filter_result_df
            filtered_df,                                       
        )

    def update_time_series_choosers(self, file):
        """
        Update the dropdowns for time series plot based on uploaded file.

        Args:
        file (UploadedFile): The uploaded file containing the data.

        Returns:
        tuple: Lists of datetime and numerical column names.
        """

        if file is None:
            return (
                gr.update(choices=[], value=None),  # date_col
                gr.update(choices=[], value=None),  # numeric_col
            )

        df = input_data(file)[0]
        data_categories = data_category(df)

        datetime_cols = data_categories.get("Datetime Columns", [])
        numeric_cols = data_categories.get("Numeric Columns", [])

        return (
            gr.update(
                choices = datetime_cols,
                value = None,
            ),
            gr.update(
                choices = numeric_cols,
                value = None,
            ),
        )
    
    def update_distribution_chooser(self, file):
        """
        Update numeric column dropdown for distribution plot.

        Args:
        file (UploadedFile): The uploaded file containing the data.

        Returns:
        gr.update: Updated dropdown component for numeric columns.
        """
        if file is None:
            return gr.update(choices=[], value=None)

        df = input_data(file)[0]
        data_categories = data_category(df)
        numeric_cols = data_categories.get("Numeric Columns", [])

        return gr.update(
            choices = numeric_cols,
            value = None,
        )

    def update_category_chooser(self, file):
        if file is None:
            return gr.update(choices=[], value=None)
    
        df = input_data(file)[0]
        data_categories = data_category(df)
    
        cat_cols = data_categories.get("Categorical Columns", [])
    
        return gr.update(
            choices=cat_cols,
            value=None
        )

    def update_scatter_choosers(self, file):
        """
        Update x/y dropdowns for scatter plot based on uploaded file.

        Args:
        file (UploadedFile): The uploaded file containing the data.

        Returns:
        tuple: Updated dropdown components for x and y numerical columns.
        """
        if file is None:
            return (
                gr.update(choices=[], value=None),  # x_col
                gr.update(choices=[], value=None),  # y_col
            )

        df = input_data(file)[0]
        data_categories = data_category(df)
        numeric_cols = data_categories.get("Numeric Columns", [])

        return (
            gr.update(
                choices = numeric_cols,
                value = None,
            ),
            gr.update(
                choices = numeric_cols,
                value = None,
            ),
        )

    def update_insight_choosers(self, file):
        """
        Update metric and group by dropdowns for insights based on uploaded file.

        Args:
        file (UploadedFile): The uploaded file containing the data.

        Returns:
        tuple: Updated dropdown components for metric and group by columns.
        """

        if file is None:
            return (
                gr.update(choices=[], value=None),              # metric_col
                gr.update(choices=["(None)"], value="(None)"),  # group_col
            )

        df = input_data(file)[0]
        data_categories = data_category(df)

        numeric_cols = data_categories.get("Numeric Columns", [])
        cat_cols = data_categories.get("Categorical Columns", [])

        metric_update = gr.update(
            choices=numeric_cols,
            value=None,
        )

        simple_cat_cols = []
        for col in cat_cols:
            try:
                nunique = df[col].nunique(dropna=True)
            except Exception:
                continue

            if nunique <= 20:
                simple_cat_cols.append(col)

        group_choices = ["(None)"] + simple_cat_cols

        group_update = gr.update(
            choices=group_choices,
            value="(None)",
        )

        return metric_update, group_update

    def launch_app(self):
        """
        Build and launch the full Gradio dashboard
        """

        with gr.Blocks(title = "Paris Airbnb Business Intelligence Dashboard") as self.demo:
            
            gr.HTML("""
                <style>
                .section-section {
                    background-color: #FBF7FF !important;   
                    padding: 40px !important;
                    border-radius: 12px !important; 
                    border: 2px solid #FCEFFF !important;   
                    margin-top: 40px !important;
                }               

                .section-section div {
                    background-color: #FAF4FF !important;   
                    box-shadow: none !important;
                }               

                .section-section button {
                    background-color: #F9F2FF !important;
                    color: #4A3A63 !important; 
                    border: 2px solid #E6D9FF !important;
                }               

                .section-section button:hover {
                    background-color: #F7EFFF !important;
                }               

                .section-section .accordion-content,
                .section-section .accordion-content .gr-box,
                .section-section .accordion-content .gr-block {
                    background-color: #F2E9FF !important;
                }
                .section-section pre,
                    
                .section-section code {
                    background-color: #F2E9FF !important;   
                    border-radius: 4px;
                }

                .section-section textarea,
                .section-section .overflow-auto,
                .section-section .overflow-scroll {
                    background-color: #F2E9FF !important;
                }
                
                </style>
                """)
            
            gr.Markdown("""
                <div style="text-align: center;">           

                <h1 style="
                    font-size: 48px; 
                    font-weight: 900; 
                    margin-bottom: 10px;
                ">          
                🎉 Airbnb Business Intelligence Dashboard 🎉
                </h1>           

                <img src="https://raw.githubusercontent.com/Stephaniekao/project/master/travelbi.jpg"
                     style="
                        width: 100%; 
                        height: 200px; 
                        object-fit: cover; 
                        border-radius: 8px;
                        margin-top: 10px;
                        margin-bottom: 20px;
                     " />           

                <h3 style="
                    font-size: 22px;
                    font-weight: 600;
                    margin-bottom: 8px;
                ">
                Dynamic, Dataset-Agnostic BI System for Market & Pricing Analysis
                </h3>           

                <p style="
                    font-size: 18px;
                    margin-top: 0px;
                ">
                Support your business decisions with data-driven insights!
                </p>            

                </div>
                """)

            gr.Markdown("\n---\n")
            gr.Markdown("## 📁 Data Upload & Validation")
            with gr.Group(elem_classes="section-section"):

                # File upload section and info display
                file_input, process_button = self.upload_file()
    
                # Data info section     
                gr.Markdown(f"## 📊 Dataset Information\n")
    
                info_box = gr.Markdown("*(Dataset info will appear here after you upload a file.)*")
                
                file_input.change(
                    fn = self.show_info,
                    inputs = file_input,
                    outputs = info_box
                )
    
                # Data preview section
                gr.Markdown("\n## 🔍 Data Preview")    
                
                with gr.Accordion("Data Preview (first/last 5 rows)", open = False):
                    preview_status = gr.Markdown("*(Dataset preview will appear here after you upload a file.)*")
    
                    preview_box = gr.DataFrame(
                        label = "Preview",
                        interactive = False,
                        visible = False
                    )
    
                file_input.change(
                    fn = self.show_preview,
                    inputs = file_input,
                    outputs = [preview_status, preview_box]
                )
    
                # data error info
                gr.Markdown("\n## ⚠️ Data Validation")   
                error_box = gr.Markdown("*(Data validation messages will appear here after you upload a file.)*")
    
                file_input.change(
                    fn = self.validate_data,
                    inputs = file_input,
                    outputs = error_box
                )
                
            gr.Markdown("\n---\n")
            gr.Markdown("## 📈 Data Exploration & Summary Statistics")
            with gr.Group(elem_classes="section-section"):
                gr.Markdown("## 📊 Automated data profiling")
                gr.Markdown("### 🔍 Column Metrics Explorer")

                with gr.Group(elem_classes="section-section"):
                    gr.Markdown("*(To be implemented: Interactive component to explore metrics for individual columns.)*")

                    numerical_selector = gr.CheckboxGroup(
                        choices = [],
                        label = "Select numerical columns",
                        interactive = True
                    )

                    categorical_selector = gr.CheckboxGroup(
                        choices = [],
                        label = "Select categorical columns",
                        interactive = True
                    )

                    metrics_button = gr.Button("Compute Metrics")

                    metrics_output = gr.Markdown("*(Metrics for selected columns will appear here.)*")

                    file_input.change(
                        fn = self.update_metrics_choosers,
                        inputs = file_input,
                        outputs = [numerical_selector, categorical_selector]
                    )

                    metrics_button.click(
                        fn = self.show_selected_metrics,
                        inputs = [file_input, numerical_selector, categorical_selector],
                        outputs = metrics_output
                    )

                auto_box = gr.Markdown("*(Dataset auto summary will appear here after you upload a file.)*")
                           
                file_input.change(
                    fn = self.show_auto,
                    inputs = file_input,
                    outputs = auto_box
                )

                gr.Markdown("## 🔥 Correlation matrix for numerical features\n")

                with gr.Tab("Correlation Heatmap"):
                    gr.Markdown("### Numerical Feature Correlation Heatmap")

                    corr_plot = gr.Plot()  

                    file_input.change(
                        fn = lambda file: plot_corr_matrix(input_data(file)[0], data_category(input_data(file)[0])),
                        inputs = file_input,
                        outputs = corr_plot
                    )   

            gr.Markdown("\n---\n")
            gr.Markdown("## ⚙️ Interactive Filtering")
            with gr.Group(elem_classes="section-section"):
            
                gr.Markdown("Use dynamic filters to narrow down the dataset by column type.")

                filter_category = gr.Dropdown(
                    choices=["Numeric Columns", "Categorical Columns", "Datetime Columns"],
                    value=None,
                    label="Select a data category",
                    interactive=True
                )

                filter_column = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Select a column",
                    interactive=True,
                    visible=False
                )

                # ---------- Numeric ----------
                numeric_status = gr.Markdown(
                    "*(Select a numeric column to see its range.)*",
                    visible=False
                )

                min_slider = gr.Slider(
                    label="Minimum Value",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=True,
                    visible=False
                )

                max_slider = gr.Slider(
                    label="Maximum Value",
                    minimum=0,
                    maximum=1,
                    value=1,
                    interactive=True,
                    visible=False
                )

                apply_numeric_btn = gr.Button(
                    "Apply Numeric Filter",
                    visible=False
                )

                # ---------- Categorical ----------
                cat_status = gr.Markdown(
                    "*(Select a categorical column to see its categories.)*",
                    visible=False
                )

                cat_values = gr.CheckboxGroup(
                    choices=[],
                    label="Select categories",
                    interactive=True,
                    visible=False
                )

                apply_cat_btn = gr.Button(
                    "Apply Categorical Filter",
                    visible=False
                )

                # ---------- Datetime ----------
                date_status = gr.Markdown(
                    "*(Select a datetime column to see its date range.)*",
                    visible=False
                )

                start_date = gr.Dropdown(
                    label="Start Date",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False
                )

                end_date = gr.Dropdown(
                    label="End Date",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False
                )

                apply_date_btn = gr.Button(
                    "Apply Datetime Filter",
                    visible=False
                )

                # ---------- Shared Result ----------
                filter_result_md = gr.Markdown(
                    "*(Filtered result summary will appear here.)*",
                    visible=False
                )
                
                filter_result_df = gr.DataFrame(
                    label="Filtered Preview",
                    interactive=False,
                    visible=False
                )

                filtered_df_state = gr.State(value=None)

                download_csv_btn = gr.DownloadButton(
                    label="⬇️ Download filtered data as CSV"
                )

                # Category change → control which components are visible and column choices
                filter_category.change(
                    fn=self.on_filter_category_change,
                    inputs=[file_input, filter_category],
                    outputs=[
                        filter_column,      # 1 Dropdown
                        numeric_status,     # 2 Numeric column description
                        min_slider,         # 3 Minimum value slider
                        max_slider,         # 4 Maximum value slider
                        apply_numeric_btn,  # 5 Apply Numeric Filter button
                        filter_result_md,   # 6 Shared result summary
                        cat_status,         # 7 Categorical column description
                        cat_values,         # 8 CheckboxGroup for selecting categories
                        apply_cat_btn,      # 9 Apply Categorical Filter button
                        filter_result_df,   #10 Shared result table
                        date_status,        #11 Datetime column description
                        start_date,         #12 Start Date picker
                        end_date,           #13 End Date picker
                        apply_date_btn,     #14 Apply Datetime Filter button
                    ]
                )

                # Column selection → if numeric, update min/max; if categorical, update options
                filter_column.change(
                    fn=self.update_numeric_range_status,
                    inputs=[file_input, filter_category, filter_column],
                    outputs=[numeric_status, min_slider, max_slider]
                )

                filter_column.change(
                    fn=self.update_categorical_values,
                    inputs=[file_input, filter_category, filter_column],
                    outputs=[cat_values, cat_status]
                )

                filter_column.change(
                    fn=self.update_datetime_range_status,
                    inputs=[file_input, filter_category, filter_column],
                    outputs=[date_status, start_date, end_date]
                )

                # Button: Apply numeric filter
                apply_numeric_btn.click(
                    fn=self.apply_numeric_filter,
                    inputs=[file_input, filter_column, min_slider, max_slider],
                    outputs=[filter_result_md, filter_result_df, filtered_df_state]
                )

                # Button: Apply categorical filter
                apply_cat_btn.click(
                    fn=self.apply_categorical_filter,
                    inputs=[file_input, filter_column, cat_values],
                    outputs=[filter_result_md, filter_result_df, filtered_df_state]
                )

                # Button: Apply datetime filter
                apply_date_btn.click(
                    fn=self.apply_datetime_filter,
                    inputs=[file_input, filter_column, start_date, end_date],
                    outputs=[filter_result_md, filter_result_df, filtered_df_state]
                )

                download_csv_btn.click(
                    fn=export_filtered_csv,
                    inputs=filtered_df_state,
                    outputs=download_csv_btn,
                )

            gr.Markdown("\n---\n")
            gr.Markdown("## 🎨 Visualizations")
            with gr.Group(elem_classes="section-section"): 

                with gr.Tabs():
                    # 🕒 Time series plot
                    with gr.Tab("🕒 Time series plot"):
                        gr.Markdown("### Trends over time")

                        date_col = gr.Dropdown(
                            label="Select time column",
                            choices=[],
                            value=None,
                            interactive=True
                        )  

                        numeric_col = gr.Dropdown(
                            label="Select numeric column",
                            choices=[],
                            value=None,
                            interactive=True
                        )  

                        agg_method = gr.Dropdown(
                            label="Aggregation",
                            choices=["mean", "sum", "median", "count"],
                            value="mean",
                            interactive=True
                        )  

                        ts_plot = gr.Plot()

                        plot_button = gr.Button("Generate Time Series Plot")   

                        ts_download_btn = gr.DownloadButton(
                            label="⬇️ Download plot as PNG"
                        )

                        file_input.change(
                            fn=self.update_time_series_choosers,
                            inputs=file_input,
                            outputs=[date_col, numeric_col],
                        )  

                        plot_button.click(
                            fn=plot_time_series,
                            inputs=[file_input, date_col, numeric_col, agg_method],
                            outputs=ts_plot,
                        )  

                        ts_download_btn.click(
                            fn=export_time_series_plot_png,
                            inputs=[file_input, date_col, numeric_col, agg_method],
                            outputs=ts_download_btn,
                        )

                    # 📊 Distribution plot
                    with gr.Tab("📊 Distribution plot"):
                        gr.Markdown("### Distribution of numeric features")
                        
                        dist_col = gr.Dropdown(
                            label="Select numeric column",
                            choices=[],
                            value=None,
                            interactive=True
                        )  

                        dist_type = gr.Dropdown(
                            label="Plot type",
                            choices=["Histogram", "Box Plot"],
                            value="Histogram",
                            interactive=True
                        )  

                        dist_plot = gr.Plot()
                        dist_button = gr.Button("Generate Distribution Plot")  

                        dist_download_btn = gr.DownloadButton(
                            label="⬇️ Download plot as PNG"
                        )
                        
                        file_input.change(
                            fn=self.update_distribution_chooser,
                            inputs=file_input,
                            outputs=dist_col,
                        )  

                        dist_button.click(
                            fn=plot_distribution,
                            inputs=[file_input, dist_col, dist_type],
                            outputs=dist_plot,
                        )  

                        dist_download_btn.click(
                            fn=export_distribution_plot_png,
                            inputs=[file_input, dist_col, dist_type],
                            outputs=dist_download_btn,
                        )

                    # 📦 Category analysis
                    with gr.Tab("📦 Category analysis"):
                        gr.Markdown("### Bar chart or pie chart")  

                        cat_col = gr.Dropdown(
                            label="Select categorical column",
                            choices=[],
                            value=None,
                            interactive=True
                        )  

                        chart_type = gr.Dropdown(
                            label="Chart Type",
                            choices=["Bar Chart", "Pie Chart"],
                            value="Bar Chart",
                            interactive=True
                        )  

                        cat_plot = gr.Plot()
                        cat_button = gr.Button("Generate Category Plot")   

                        cat_download_btn = gr.DownloadButton(
                            label="⬇️ Download plot as PNG"
                        )

                        file_input.change(
                            fn=self.update_category_chooser,
                            inputs=file_input,
                            outputs=cat_col
                        )  

                        cat_button.click(
                            fn=plot_categorical_distribution,
                            inputs=[file_input, cat_col, chart_type],
                            outputs=cat_plot
                        )

                        cat_download_btn.click(
                            fn=export_categorical_plot_png,
                            inputs=[file_input, cat_col, chart_type],
                            outputs=cat_download_btn
                        )

                    with gr.Tab("🔗 Scatter plot"):
                        gr.Markdown("### Scatter plot: Relationship between two numeric variables")                 

                        x_col = gr.Dropdown(
                            label="X-axis (numeric column)",
                            choices=[],
                            value=None,
                            interactive=True,
                        )                   

                        y_col = gr.Dropdown(
                            label="Y-axis (numeric column)",
                            choices=[],
                            value=None,
                            interactive=True,
                        )                   

                        scatter_plot = gr.Plot()
                        scatter_button = gr.Button("Generate Scatter Plot")                 

                        scatter_download_btn = gr.DownloadButton(
                            label="⬇️ Download plot as PNG"
                        )

                        file_input.change(
                            fn=self.update_scatter_choosers,
                            inputs=file_input,
                            outputs=[x_col, y_col],
                        )                   

                        scatter_button.click(
                            fn=plot_scatter,
                            inputs=[file_input, x_col, y_col],
                            outputs=scatter_plot,
                        )

                        scatter_download_btn.click(
                            fn=export_scatter_plot_png,
                            inputs=[file_input, x_col, y_col],
                            outputs=scatter_download_btn
                        )

            gr.Markdown("\n---\n")
            gr.Markdown("## 💡 Automatic Insights")
            with gr.Group(elem_classes="section-section"): 

                gr.Markdown("## 📌 Top/Bottom performers")

                metric_col = gr.Dropdown(
                    label="Select metric (numeric column)",
                    choices=[],
                    value=None,
                    interactive=True,
                )

                group_col = gr.Dropdown(
                    label="Group by (optional, categorical)",
                    choices=[],
                    value="(None)",
                    interactive=True,
                )

                top_n = gr.Slider(
                    label="How many Top/Bottom items to show",
                    minimum=3,
                    maximum=20,
                    value=5,
                    step=1,
                    interactive=True,
                )

                insights_md = gr.Markdown("*(Insights will appear here.)*")
                generate_btn = gr.Button("Generate Insights")

                file_input.change(
                    fn=self.update_insight_choosers,
                    inputs=file_input,
                    outputs=[metric_col, group_col],
                )

                generate_btn.click(
                    fn = generate_insight,
                    inputs = [file_input, metric_col, group_col, top_n],
                    outputs = insights_md,
                )
                
                gr.Markdown("\n---\n")
                gr.Markdown("## 🧠 Basic trends or anomalies")

                auto_insights_md = gr.Markdown(
                    "*(Automatic trend and anomaly insights will appear here.)*"
                )

                file_input.change(
                    fn = generate_trends_and_anomalies_from_df,
                    inputs = file_input,
                    outputs = auto_insights_md,
                )

            self.demo.launch(share = True)
