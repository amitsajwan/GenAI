
Prompt 1 (Concise): PreliminaryAnalysisAgent

Goal:
Generate preliminary_script.py to load raw stock data, perform initial inspection, standardize date information (creating and setting numeric_date_idx as index while keeping original 'Date' column), and save an initial analysis report and the processed DataFrame.

Inputs (Agent will receive):

raw_data_path: Path to the raw stock data CSV.
(Optional) schema_description_content: Text content describing data schema.
Script (preliminary_script.py) Instructions:

Load & Inspect: Load data from raw_data_path. Print shape, info, head, descriptive stats (all columns), initial missing value counts, unique value counts per column.
Date Handling & Indexing (CRITICAL):
Identify and parse the primary 'Date' column to datetime.
Sort DataFrame chronologically by 'Date'.
Create numeric_date_idx (e.g., Unix timestamp seconds from 'Date').
Set numeric_date_idx as index. Preserve original 'Date' column.
Outputs & Validation (within the script):
Save DataFrame: To initially_processed_data.csv (with numeric_date_idx as index, use index=True).
Save Report: All printed outputs/findings to preliminary_analysis_report.txt.
Validate:
Confirm initially_processed_data.csv & preliminary_analysis_report.txt exist.
Load initially_processed_data.csv, check its shape, confirm numeric_date_idx is index, and 'Date' column exists.
Print validation status.


Prompt 2 (Concise): EDAAgent (EDA & Data Cleaning Agent)

Goal:
Generate eda_script.py to perform detailed EDA and cleaning. Leverage embedded {preliminary_analysis_report_content} and data from initially_processed_data.csv. Focus EDA on the training portion. Chronologically split data into train/val/test, save them, and save a detailed EDA summary.

Inputs (Agent will receive):

processed_data_path: Path to initially_processed_data.csv.
{preliminary_analysis_report_content}: Embedded text content.
train_ratio, val_ratio, test_ratio: e.g., 0.7, 0.15, 0.15.
Script (eda_script.py) Instructions:

Load Data & Use Preliminary Insights: Load data from processed_data_path (ensure numeric_date_idx is index). Use embedded {preliminary_analysis_report_content} to guide cleaning (e.g., type corrections).
In-Depth EDA (on Training Portion):
Based on train_ratio, conceptually define training data.
Generate and save key visualizations (or describe textually for report): distributions, time-series plots (Close, Volume), correlations for this portion.
Summarize findings: trends, seasonality, volatility, key relationships.
Chronological Data Splitting: Split the full cleaned DataFrame into train_df, val_df, test_df using ratios.
Outputs & Validation (within the script):
Save Split Datasets: Cleaned_Train.csv, Cleaned_Val.csv, Cleaned_Test.csv. CRITICAL: Use index=True to preserve numeric_date_idx.
Save EDA Report: eda_result_summary.txt (detailed findings, visualization summaries, cleaning actions, split details, date ranges).
Validate:
Check Cleaned_*.csv files and eda_result_summary.txt exist.
Load each Cleaned_*.csv; verify numeric_date_idx is index, shapes align with ratios, 'Date' & 'Close' columns exist.
Print validation status.



Prompt 3 (Concise): FeatureEngineeringAgent

Goal:
Generate feature.py for dynamic and comprehensive feature engineering. Use embedded {preliminary_analysis_report_content} and {eda_result_summary_content} with Cleaned_*.csv data. Dynamically propose and create relevant market signals. Follow this strict order: initial imputation, new signal creation, NaN handling for new signals (row drops, ensuring y-alignment), target separation, then final imputation/scaling/encoding for X features. Save all processed X/y sets, fitted transformers, and a detailed FE report. Preserve numeric_date_idx throughout.

Inputs (Agent will receive):

Paths: Cleaned_Train.csv, Cleaned_Val.csv, Cleaned_Test.csv.
{preliminary_analysis_report_content}: Embedded text.
{eda_result_summary_content}: Embedded text.
target_column_name: 'Close'.
date_index_name: 'numeric_date_idx'.
Script (feature.py) Key Operational Sequence:

Load Data: Cleaned_Train.csv, etc. (set date_index_name as index).
Initial Imputation (Entire DataFrames): Fit imputers (numerical & categorical, e.g., median/ffill & most_frequent) ONLY on train_df. Save as initial_*.pkl. Transform train_df, val_df, test_df.
Dynamic Market Signal Generation:
Analyze embedded reports & data: Propose, justify, and create relevant market signals (MAs, Lags, RSI, MACD, Volatility, etc.) with appropriate parameters (window sizes, etc.). Append to DataFrames.
Log created signals/parameters for the FE report.
Post-Signal NaN Handling: Drop rows with NaNs from DataFrames (that still include target) due to signal generation. This ensures X and y alignment.
Separate Target: From NaN-handled DataFrames, separate target_column_name into y_train, y_val, y_test. Remainder forms X_train, X_val, X_test.
Final Imputation (X features only): Fit imputers ONLY on X_train. Save as final_*.pkl. Transform X_train, X_val, X_test.
Feature Scaling (Numerical X features): Fit StandardScaler ONLY on X_train. Save as scaler.pkl. Transform X sets.
Categorical Encoding (Categorical X features): Fit OneHotEncoder (handle_unknown='ignore') ONLY on X_train. Save as one_hot_encoder.pkl. Transform X sets. Combine all processed X features, preserving index.
Outputs & Validation (within the script):
Save Transformers: All specified .pkl files.
Save Datasets: x_train.csv, y_train.csv, etc. (all with index=True for date_index_name).
Save FE Report: feature_engineering_report.txt (details: dynamic feature choices & params, all processing steps, saved transformers, final feature list in x_train.csv, final data shapes).
Validate:
Check all .pkl & .csv files exist.
Load x_*.csv, verify index, check for NaNs (should be none/few), check shapes. Load y_*.csv, verify shape alignment.
Print validation status.



Prompt 4 (Concise): ModelTrainingAgent

Goal:
Generate model_training.py to select, train, and tune a regression model. Use embedded {preliminary_analysis_report_content}, {eda_result_summary_content}, and crucially {feature_engineering_report_content} with x_*.csv and y_*.csv data. Perform hyperparameter tuning using the validation set (with TimeSeriesSplit). Save the final model and report validation metrics.

Inputs (Agent will receive):

Paths: x_train.csv, y_train.csv, x_val.csv, y_val.csv.
{preliminary_analysis_report_content}: Embedded text.
{eda_result_summary_content}: Embedded text.
{feature_engineering_report_content}: Embedded text.
date_index_name: 'numeric_date_idx'.
Script (model_training.py) Instructions:

Load Data: x_train.csv, y_train.csv, etc. (set date_index_name as index).
Dynamic Model Selection:
Analyze embedded reports (especially {feature_engineering_report_content} for feature characteristics).
Propose 1-2 suitable regression model types (e.g., XGBoost, LightGBM) with justification.
Hyperparameter Tuning:
For chosen model(s), define search space. Use GridSearchCV or RandomizedSearchCV.
CRITICAL: Use TimeSeriesSplit from sklearn.model_selection as the cv strategy.
Use regression scoring (e.g., 'neg_root_mean_squared_error'). Fit using x_train and y_train.
Train Final Model: With best HPs found, train on entire x_train, y_train.
Evaluate on Validation Set: Predict on x_val and report RMSE, MAE, R² against y_val.
Outputs & Validation (within the script):
Save Model: trained_model.pkl (using joblib).
Save Training Report: model_training_report.txt (model choice & justification, HPs, validation metrics, path to saved model).
Validate:
Check trained_model.pkl & model_training_report.txt exist.
Load trained_model.pkl to verify.
Print validation status. 
