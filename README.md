### Airbnb Business Intelligence Dashboard ###
# Pricing Insights & Market Competition Analysis #
# Dynamic, Dataset-Agnostic BI System for Market & Pricing Analysis #

This project builds an interactive Business Intelligence (BI) dashboard for analyzing Airbnb market competition.

It is fully dynamic and supports any Airbnb-style dataset (from Paris, New York, Tokyo, or any city), thanks to automated data-type detection and adaptive visualizations.

#　📦 Dataset Source

You can upload any CSV or Excel file containing Airbnb-like listing data.
The system automatically detects:
- Numerical columns
- Categorical columns
- Datetime columns
- Missing values
- Outliers
- Correlations

This allows the dashboard to work dynamically with different dataset schemas, cities, and file formats.

1️⃣ ⭐ Recommended – Paris Airbnb Listings
Original testing dataset for this project.
🔗 https://www.kaggle.com/datasets/abaghyangor/airbnb-paris

2️⃣ ⭐ Stability Tes: New York City Airbnb Open Data
Used to verify that the dashboard remains stable even when certain data types (such as datetime columns) are missing.
🔗 https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

These datasets are just examples.
The dashboard is not limited to the samples above—
you may upload any Airbnb-style dataset, and the system will automatically adapt.

⚠️ About Missing Column Types
Some visualizations or filters may appear disabled or show empty dropdown options.
This is expected behavior when the uploaded dataset does not contain the required column types.

Examples:
- No datetime columns → Time-series plots are disabled
- No categorical columns → Category charts and filters are disabled
- Only one numeric column → Scatter plot is not available

The dashboard is fully dynamic and automatically adapts to the dataset.
It never crashes or throws errors when the data schema changes.

# 👥 Intended Users

This dashboard is designed for:

- Travel Agencies  
- Resort / Hotel Managers  
- Airbnb Hosts  
- OTA Platforms (Booking / Agoda)  
- Tourism Boards  
- Data Analysts / BI Analysts  

# 🎯 Objectives

- Provide a market competition analysis tool for tourism and accommodation industries.
- Deliver a **dynamic**, **dataset-agnostic** BI dashboard with automated analysis.
- Allow users to:
  - explore pricing
  - compare room types
  - analyze guest capacity
  - inspect availability trends
  - evaluate host behaviors
  - compare neighborhood performance
- Enable users to upload various Airbnb datasets without modifying the source code.

# 💼 Business Value

Organizations can answer:

- Which areas are most competitive or profitable?
- How do different room types compare in pricing and occupancy?
- Do professional hosts perform better?
- What seasonal changes can be observed?
- How do price and review trends evolve over time?

The dashboard provides:

- Market price benchmarks  
- Neighborhood competitiveness  
- Host performance insights  
- Seasonal and trend analysis  
- Data-driven recommendations  

# 🧩 Dashboard Features

## ✔ Upload Airbnb datasets (CSV / Excel)
- Automatically detects structure  
- Validates data integrity  
- Displays dataset preview & issues  

## ✔ Automated Data Profiling
- Numerical summary statistics  
- Categorical summary statistics  
- Missing & duplicate value analysis  
- Dynamic data type detection  
- Correlation matrix heatmap  

## ✔ Interactive Filtering System
- Numerical range filters  
- Categorical multi-select filters  
- Datetime range filters  
- Real-time filtered preview  
- Export filtered results (.csv)  

## ✔ Visualizations (Fully Dynamic)
1. **Time Series Plot**
   - Trends over time using any datetime field
   - Aggregation: mean / sum / median / count

2. **Distribution Plots**
   - Histogram  
   - Box plot  

3. **Category Analysis**
   - Bar chart or pie chart  
   - For any categorical field  

4. **Scatter Plot**
   - Two numerical columns  
   - Explore relationships  

5. **Correlation Heatmap**
   - Automatically generated based on numerical features  

## ✔ Automated Insights
The system extracts insights such as:

- Top & bottom performers  
- Pricing anomalies  
- Outlier detection (IQR method)  
- Trends in numerical features  
- Category-level performance differences  
- Neighborhood price ranking  
- Host performance gaps  

These are computed dynamically for any dataset.

# 🚀 Project Status

This README defines the project direction before full implementation.
The dashboard will be developed using:

- pandas
- Gradio
- matplotlib / seaborn / plotly
- Python 3.8+

# 📁 Project Structure (Planned)
project/
- app.py
- data_processor.py
- visualizations.py
- insights.py
- utils.py
- data/
    - sample1.csv  (Paris listings)
    - sample2.csv  (Other city listings)
- README.md

# 🔥 Why This Dashboard Is Unique

Unlike traditional dashboards, this one:

- Does not require predefined columns  
- Adapts dynamically to any dataset  
- Automatically generates UI controls  
- Automatically detects column types  
- Provides insight generation without configuration  

This makes it highly reusable and flexible for different cities & datasets.

# 🚀 Project Status

✔ Core dashboard completed  
✔ Automated insights implemented  
✔ Dynamic column detection completed  
✔ All visualizations fully functional  
✔ Supports any dataset uploaded by user  

Future improvements may include:

- Cross-city comparison view  
- Machine learning pricing predictions  
- Geo-visualization (map-based insights)  
- PDF/HTML report export  

# 📜 License

This project uses publicly available datasets from Kaggle.  
All rights belong to the original dataset creators.




