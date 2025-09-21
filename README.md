# Comprehensive Analysis of Texas Climate Patterns using Distributed Computing with Apache Spark

This repository contains a comprehensive analysis of historical Texas weather data, demonstrating an end-to-end data engineering and machine learning pipeline. The project leverages **Apache Spark** for scalable, distributed processing of large-scale climate data and the Python data science ecosystem for advanced statistical analysis, machine learning, and visualization.

***

## Project Overview

The primary goal of this project is to quantitatively validate common claims about Texas's diverse climate using decades of weather station data. The secondary goal is to build a machine learning classifier to predict a station's [Köppen climate classification](https://en.wikipedia.org/wiki/K%C3%B6ppen_climate_classification) based on its weather patterns.

This project was designed to handle a dataset too large for single-machine processing, making Apache Spark an essential tool for all initial data ingestion, cleaning, transformation, and feature engineering steps.

### Key Analyses Conducted:
* **Regional Climate Profiling:** Calculating and comparing aggregate statistics (mean, variance) for temperature and precipitation across five distinct Texas climate regions.
* **Hypothesis Testing:** Statistically testing claims, such as the difference in diurnal temperature range between the arid west and humid east.
* **Seasonal Pattern Recognition:** Analyzing monthly data to identify and interpret seasonal signatures, such as the West Texas monsoon and the Gulf Coast hurricane season.
* **Extreme Weather Event Analysis:** Isolating and analyzing the frequency and distribution of events like tornadoes (WT10) and days with extreme rainfall ($\ge 100$ mm).
* **Principal Component Analysis (PCA):** Applying PCA to distill complex, high-dimensional climate data into its core components, revealing the dominant climate gradients across the state.


***

## Machine Learning: Köppen Climate Classification

A key component of this project was to develop a supervised machine learning model to classify weather stations into their respective Köppen climate zones (e.g., 'Cfa', 'BSh', 'BSk').

* **Objective**: Predict the Köppen label based on yearly weather patterns for temperature (`TOBS`), snowfall (`SNWD`), and precipitation (`PRCP`).
* **Feature Engineering**:
    1.  Raw, byte-encoded annual time-series data for each measurement was processed in a distributed manner using a custom function on the **Spark RDD**.
    2.  **Principal Component Analysis (PCA)** was applied to each measurement's time-series to reduce its dimensionality from 366 daily values to 3 principal components, creating a concise feature vector.
    3.  The PCA features for all measurements were concatenated to form the final feature matrix.
* **Modeling**: An **XGBoost classifier** was trained and tuned to perform the multi-class classification. The model's performance was evaluated using two approaches: the native XGBoost API with early stopping and the Scikit-learn wrapper.
* **Result**: The final model successfully learned the patterns differentiating the climate zones, with the results evaluated using accuracy and a confusion matrix.


***

## Technical Architecture & Workflow

The architecture of this project is centered around a **hybrid distributed/single-node pipeline**, leveraging the unique strengths of Apache Spark for large-scale ETL and the rich functionality of Python libraries for specialized modeling.

1.  **Data Ingestion (Spark)**: Raw weather and station data, stored in the efficient **Parquet** format, are loaded into Spark DataFrames.
2.  **Distributed ETL & Transformation (Spark)**:
    * A key challenge was the packed format of the raw data (a year of daily measurements in a single field). This was solved using a scalable two-step process:
        1.  A Python **User-Defined Function (UDF)** was developed to decode, clean, and scale the raw byte-encoded values into numerical arrays.
        2.  Spark's powerful `posexplode` function was used to efficiently transform the data from a wide format to a tidy, long format (one row per station-day), enabling distributed analysis.
    * Geographic regions were engineered as a new feature using Spark's DataFrame API (`withColumn` and `when`).
3.  **Large-Scale Aggregation (Spark)**: All large-scale aggregations—calculating daily, monthly, and regional statistics—were performed using Spark's highly optimized `groupBy`, `agg`, and `pivot` operations.
4.  **Data Persistence**: To optimize the Spark computational graph (DAG) and create efficient checkpoints, intermediate transformed and aggregated DataFrames were persisted back to Parquet.
5.  **Downstream Analysis & ML (Pandas & Scikit-learn)**: For advanced statistics, visualization, and machine learning, the aggregated, now smaller, Spark DataFrames were collected to the driver node as Pandas DataFrames. This hybrid approach uses the right tool for the job: Spark for the heavy lifting and Scikit-learn/XGBoost for sophisticated modeling.


***

## Key Technical Highlights & Skills Demonstrated

This project showcases a strong, practical understanding of distributed data processing and data science principles.

* **Distributed Computing**: **Apache Spark**, **PySpark**.
* **Big Data Engineering**:
    * Designing and implementing robust ETL pipelines for large-scale datasets.
    * Proficiently handling complex data structures and formats.
    * Writing and applying **UDFs** for custom data transformations.
    * Data reshaping and normalization at scale using functions like `posexplode`.
* **Spark API Proficiency**:
    * Expert use of the **Spark DataFrame API** for filtering, joining, and feature engineering.
    * Advanced aggregation techniques using `groupBy`, `agg`, and `pivot`.
    * Experience with the lower-level **Spark RDD API** for custom, row-level transformations.
* **Performance Optimization**:
    * Leveraging the **Parquet** file format for efficient, columnar storage and I/O.
    * Using `.cache()` and persisting intermediate DataFrames to optimize Spark execution plans.
* **Machine Learning**:
    * End-to-end pipeline development from raw data to a predictive model.
    * Advanced feature engineering using **PCA** for dimensionality reduction.
    * Model training, hyperparameter tuning, and evaluation using **XGBoost** and **Scikit-learn**.
* **Data Analysis & Visualization**:
    * Applying statistical tests (t-test) to validate hypotheses.
    * Conducting time-series, seasonal, and spatial analysis.
    * Creating insightful visualizations (boxplots, line charts, heatmaps, CDF plots) using **Matplotlib** and **Seaborn**.
