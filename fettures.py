1. Prompt for PreliminaryAnalysisAgent

Goal:
Generate a Python script (preliminary_script.py) to load raw stock data, perform initial inspection, standardize date information (create and set numeric_date_idx as index, keep original 'Date' column), and save an analysis report text and the initially processed DataFrame CSV.

Inputs (Agent will receive):

raw_data_path: Path to the raw stock data CSV (e.g., raw_stock_data.csv).
Script (preliminary_script.py) Instructions:

Load & Inspect: Load data. Print shape, .info(), .head(), .describe(include='all'), missing value counts (.isnull().sum()), unique value counts (.nunique()).
Date Handling & Indexing:
Identify and parse the 'Date' column to datetime.
Sort DataFrame chronologically by 'Date'.
Create numeric_date_idx (Unix timestamp seconds from 'Date').
Set numeric_date_idx as index. Retain 'Date' (datetime) as a regular column.
Outputs & Validation:
Save DataFrame: To initially_processed_data.csv (use index=True).
Save Report: All printed inspection outputs to preliminary_analysis_report.txt.
Validate (in script): Check file creation (initially_processed_data.csv, preliminary_analysis_report.txt). Load initially_processed_data.csv, verify index name is numeric_date_idx, 'Date' column exists, and shape. Print validation status.

2. Prompt for EDAPromptGeneratorAgent

Goal:
You are an expert AI assistant. Your task is to generate a detailed and actionable prompt string. This prompt string will be used by a subsequent EDAAgent to guide its generation of an eda_script.py. Base your generated prompt only on the findings within the provided {preliminary_analysis_report_content}.

Input (for you, EDAPromptGeneratorAgent):

{preliminary_analysis_report_content}: Full text content from the PreliminaryAnalysisAgent.
Output (What you must generate - A single prompt string for the EDAAgent):
The prompt string you generate for the EDAAgent must instruct it to create eda_script.py with the following capabilities:

Objective: Perform detailed EDA and cleaning on initially_processed_data.csv, leveraging insights from {preliminary_analysis_report_content} (which you will embed in the prompt you are generating). Then, chronologically split data into train/val/test (e.g., 0.7/0.15/0.15 ratios), save these Cleaned_Train/Val/Test.csv files (preserving numeric_date_idx as index), and produce a detailed eda_result_summary.txt.
Key Instructions for EDAAgent's script (ensure these points are covered in your generated prompt):
Load initially_processed_data.csv (confirming numeric_date_idx as index).
Crucially, use specific findings from the embedded {preliminary_analysis_report_content} to guide cleaning. (e.g., if report mentions column 'X' has Y% missing values, the EDA script should specifically address 'X'; if 'Y' has dtype issues, script should correct 'Y').
Perform in-depth EDA on the conceptual training portion (visualizations or textual descriptions, correlations, trends).
Chronologically split the full, cleaned DataFrame.
Save Cleaned_Train.csv, Cleaned_Val.csv, Cleaned_Test.csv (with index=True).
Save eda_result_summary.txt (detailed EDA findings, actions taken).
The eda_script.py must validate its own outputs (file existence, data integrity of CSVs).
3. Prompt for FeatureEngineeringAgent

Goal:
Generate feature.py for dynamic and comprehensive feature engineering. Use embedded reports ({preliminary_analysis_report_content}, {eda_result_summary_content}) and Cleaned_*.csv data. Dynamically propose and create relevant market signals based on analysis. Strictly follow this order: initial imputation, new signal creation, handle NaNs from new signals (row drops, ensure y-alignment), then separate target ('Close'). Finally, apply final imputation/scaling/encoding to X features (fit ONLY on train, transform all, save transformers). Save all outputs and a detailed feature_engineering_report.txt. Preserve numeric_date_idx.

Inputs (Agent will receive):

Paths to Cleaned_Train.csv, Cleaned_Val.csv, Cleaned_Test.csv.
{preliminary_analysis_report_content}: Embedded text.
{eda_result_summary_content}: Embedded text.
target_column_name: 'Close'.
date_index_name: 'numeric_date_idx'.
Script (feature.py) Key Operational Sequence (Instruct agent to implement these):

Load Data: (From Cleaned_*.csv, set date_index_name as index).
Initial Imputation (Entire DFs): Fit imputers (num: median/ffill; cat: most_frequent) ONLY on train_df. Save (initial_*.pkl). Transform all.
Dynamic Market Signal Generation: Analyze embedded reports & data. Propose, justify, parameterize, and create market signals (MAs, Lags, RSI, etc.). Append. Log created signals for report.
Post-Signal NaN Handling: Drop rows with NaNs from DFs (still including target) due to new signals.
Separate Target: From NaN-handled DFs, separate target_column_name into y_train/val/test. Remainder is X_train/val/test.
Final Imputation (X features only): Fit imputers ONLY on X_train. Save (final_*.pkl). Transform X_train/val/test.
Feature Scaling (Numerical X): Fit StandardScaler ONLY on X_train. Save (scaler.pkl). Transform X sets.
Categorical Encoding (Categorical X): Fit OneHotEncoder (handle_unknown='ignore') ONLY on X_train. Save (one_hot_encoder.pkl). Transform X sets. Combine all X features, preserve index.
Outputs & Validation (within the script):
Save Transformers: All specified .pkl files.
Save Datasets: x_train.csv, y_train.csv, etc. (all with index=True for date_index_name).
Save FE Report: feature_engineering_report.txt (dynamic feature choices, params, all processing, transformers, final feature list, shapes).
Validate: Check all files exist. Load x_*.csv, verify index, no NaNs, shapes. Load y_*.csv, verify shape alignment. Print validation status.
4. Prompt for ModelTrainingAgent

Goal:
Generate model_training.py to select, train, and tune a regression model. Use embedded reports ({preliminary_analysis_report_content}, {eda_result_summary_content}, {feature_engineering_report_content}) and x_*.csv, y_*.csv data. Perform hyperparameter tuning using TimeSeriesSplit on the validation approach. Save the final model and report validation metrics.

Inputs (Agent will receive):

Paths to x_train.csv, y_train.csv, x_val.csv, y_val.csv.
{preliminary_analysis_report_content}: Embedded text.
{eda_result_summary_content}: Embedded text.
{feature_engineering_report_content}: Embedded text.
date_index_name: 'numeric_date_idx'.
Script (model_training.py) Instructions:

Load Data: (From x_*.csv, y_*.csv, set date_index_name as index).
Dynamic Model Selection: Analyze embedded reports (especially FE report for feature characteristics). Propose 1-2 suitable regression models (e.g., XGBoost, LightGBM) with brief justification.
Hyperparameter Tuning: For chosen model(s), define search space. Use GridSearchCV or RandomizedSearchCV with TimeSeriesSplit (from sklearn.model_selection) as cv strategy. Use regression scoring (e.g., 'neg_root_mean_squared_error'). Fit using x_train and y_train.
Train Final Model: With best HPs, train on entire x_train, y_train.
Evaluate on Validation Set: Predict on x_val and report RMSE, MAE, R² against y_val.
Outputs & Validation (within the script):
Save Model: trained_model.pkl (using joblib).
Save Training Report: model_training_report.txt (model choice & justification, HPs, validation metrics, path to saved model).
Validate: Check trained_model.pkl & model_training_report.txt exist. Load trained_model.pkl to verify. Print validation status.
 
