preliminary_analysis_prompt = """
**Goal:**
Generate a Python script that performs initial loading, inspection, and date standardization for raw stock data, then saves a report and the processed DataFrame.

**Inputs:**
* `raw_data_path`: String, path to the raw stock data CSV file.

**Output Script Name:** `preliminary_script.py`

**Python Script Instructions (for `preliminary_script.py`):**

1.  **Load & Inspect Data:**
    * Load the CSV from `raw_data_path` into a pandas DataFrame.
    * Collect for reporting: DataFrame shape, `.info()` output, `.head(5)` output, `.describe(include='all')` output, initial missing value counts per column (`.isnull().sum()`), and unique value counts per column (`.nunique()`).
2.  **Date Handling & Indexing:**
    * Identify and parse the primary 'Date' column to pandas datetime objects. Handle parsing issues gracefully (e.g., by coercing errors to `NaT` and dropping corresponding rows) and note them for the report.
    * Sort the DataFrame chronologically by this 'Date' column.
    * Create a new column named `numeric_date_idx` from the 'Date' column (e.g., Unix timestamp in seconds).
    * Set `numeric_date_idx` as the DataFrame's index.
    * Ensure the original 'Date' column (as datetime objects) is retained within the DataFrame.
3.  **Prepare Report Content:** Compile all collected inspection details (from step 1) and date handling actions (from step 2) into a single string.

**Python Script Outputs (Files to be created):**
1.  `initially_processed_data.csv`: The processed DataFrame with the 'Date' column and `numeric_date_idx` as index (saved with `index=True`).
2.  `preliminary_analysis_report.txt`: Text file containing the report content string from step 3 of "Python Script Instructions".

**Python Script Output Validation (Logic to be included at the end of `preliminary_script.py`):**
1.  Verify that `initially_processed_data.csv` and `preliminary_analysis_report.txt` were created.
2.  Load `initially_processed_data.csv` into a temporary DataFrame.
3.  Check:
    * Its shape is as expected (non-empty).
    * Its index is named `numeric_date_idx`.
    * The 'Date' column exists.
4.  Print a status message: "Preliminary analysis outputs validated successfully." or appropriate error messages if checks fail.
"""

eda_agent_prompt = """
**Goal:**
Generate a Python script (`eda_script.py`) to perform detailed EDA and cleaning
on stock data.

**Overall Objective:**
Perform detailed EDA and cleaning on `initially_processed_data.csv` (path to be provided as an input to eda_script.py).
The script MUST leverage specific insights from the embedded Preliminary Analysis Report below.
After cleaning and EDA, it must chronologically split the data (e.g., 0.7 train, 0.15 val, 0.15 test ratios)
into `Cleaned_Train.csv`, `Cleaned_Val.csv`, and `Cleaned_Test.csv` (preserving `numeric_date_idx` as index and ensuring
'Close' and other OHLCV columns are included). A detailed `eda_result_summary_content` string (also saved as `eda_result_summary.txt`)
must be produced.

**Python Script Inputs (for `eda_script.py`):**
* `processed_data_path`: String, path to `initially_processed_data.csv`.
* `preliminary_analysis_report_content`: String, the full text content from the Preliminary Analysis Report provided below.
* `train_ratio`: Float, e.g., 0.7.
* `val_ratio`: Float, e.g., 0.15.
* `test_ratio`: Float, e.g., 0.15. (Note: The sum of these should be 1.0)
* `target_column`: String, 'Close'.
* `ohlcv_columns`: List of Strings, ['Open', 'High', 'Low', 'Close', 'Volume'].
* `date_index_name`: String, 'numeric_date_idx'.

**Python Script Instructions (for `eda_script.py`):**

1.  **Load Data:**
    * Load `processed_data_path` into a pandas DataFrame, confirming `numeric_date_idx` as the index.

2.  **Perform Targeted Data Cleaning:**
    * Analyze the `preliminary_analysis_report_content` provided below to identify specific data quality issues
        (e.g., missing values, incorrect data types, or anomalies in specific columns like 'Date' or OHLCV data).
    * Implement cleaning steps based on these identified issues. For example:
        * If the report indicates parsing issues with the 'Date' column, ensure rows with `NaT` (Not a Time) are handled (e.g., dropped if they are few and represent bad data, or imputed if appropriate for the domain and many). The `PreliminaryAnalysisAgent` already handled this by dropping rows with `NaT` in 'Date'. Confirm that no `NaT` values exist in the date-related columns after loading.
        * Address any missing values identified in the `preliminary_analysis_report_content` for OHLCV or other critical columns. For time series, forward-fill or backward-fill might be appropriate, or dropping rows/columns if missingness is extensive and critical. Prioritize imputation for 'Close' and 'Volume' if missing.
        * Check for and handle any obvious outliers or erroneous values in OHLCV columns (e.g., negative prices, excessively high volumes, or 'Low' being greater than 'High'). Use descriptive statistics and visualizations to identify them. Consider capping or removing extreme outliers if they are clearly data entry errors.
        * Ensure all OHLCV columns are numeric. Convert if necessary.

3.  **Conduct In-depth EDA on Conceptual Training Portion:**
    * Before splitting, or after a preliminary split for EDA purposes (but not for final saved files), analyze distributions, correlations, and time-series patterns for 'Close' and 'Volume'.
    * **Generate key visualizations or provide detailed textual descriptions for the `eda_result_summary_content`:**
        * **Distributions:** Histograms/KDE plots for 'Close', 'Volume', 'Open', 'High', 'Low' (and potentially other numeric columns) to observe their distributions, skewness, and potential outliers.
        * **Time-series plots:** Plot 'Close' and 'Volume' over time to observe trends, seasonality (if any), and volatility.
        * **Correlations:** Heatmap of correlations between OHLCV columns.
        * **Missing Value Visualization:** A bar chart or heatmap showing missing values across columns after cleaning.
        * **Outlier Detection (Textual):** Describe any identified outliers and the strategy chosen to handle them.
    * Summarize findings for the `eda_result_summary_content`.

4.  **Chronologically Split Data:**
    * Split the *full, cleaned* DataFrame into training, validation, and test sets based on the provided ratios (`train_ratio`, `val_ratio`, `test_ratio`). Maintain chronological order.
    * Ensure 'Close' and `ohlcv_columns` are present in all split DataFrames.

**Python Script Outputs (Files to be created):**
1.  `Cleaned_Train.csv`: Training DataFrame (with `index=True` for `numeric_date_idx`).
2.  `Cleaned_Val.csv`: Validation DataFrame (with `index=True` for `numeric_date_idx`).
3.  `Cleaned_Test.csv`: Test DataFrame (with `index=True` for `numeric_date_idx`).
4.  `eda_result_summary.txt`: Text file containing the detailed `eda_result_summary_content` string.

**Python Script Output Validation (Logic to be included at the end of `eda_script.py`):**
1.  Verify that `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv`, and `eda_result_summary.txt` were created.
2.  Load each `Cleaned_*.csv` into temporary DataFrames.
3.  Check:
    * `numeric_date_idx` is the index for all loaded DataFrames.
    * Shapes align with the split ratios (approximately).
    * 'Close' and all `ohlcv_columns` exist in each DataFrame.
    * There are no `NaN` values in the 'Close' column of any split dataset.
4.  Print a status message: "EDA outputs validated successfully." or appropriate error messages if checks fail.

**Preliminary Analysis Report Content (Embedded):**

{preliminary_analysis_report_content}

"""

feature_engineering_agent_prompt = """
**Goal:**
Generate a Python script (`feature.py`) to perform dynamic and comprehensive feature engineering.

**Overall Objective:**
Perform dynamic and comprehensive feature engineering using `Cleaned_*.csv` data (paths to be provided).
The script MUST leverage insights from the embedded Preliminary Analysis Report and EDA Summary below
to dynamically propose, justify, and create relevant market signals.
It must follow a strict order:
1. Initial imputation (on full DataFrames, including 'Close').
2. New market signal creation (using 'Close' and other OHLCV from full, imputed DataFrames).
3. Handle NaNs from new signals (row drops on full DataFrames, ensuring y-alignment later).
4. THEN, separate target ('Close').
5. Finally, apply final imputation/scaling/encoding to X features (fit ONLY on train, transform all, save transformers).
All outputs (processed X/Y datasets, fitted transformers) must be saved, along with a detailed
`feature_engineering_report_content` string (also saved as `feature_engineering_report.txt`).
`numeric_date_idx` MUST be preserved.

**Python Script Inputs (for `feature.py`):**
* `cleaned_train_path`: String, path to `Cleaned_Train.csv`.
* `cleaned_val_path`: String, path to `Cleaned_Val.csv`.
* `cleaned_test_path`: String, path to `Cleaned_Test.csv`.
* `preliminary_analysis_report_content`: String, the full text content from the Preliminary Analysis Report provided below.
* `eda_result_summary_content`: String, the full text content from the EDA Result Summary provided below.
* `target_column_name`: String, 'Close'.
* `date_index_name`: String, 'numeric_date_idx'.

**Python Script Key Operational Sequence (for `feature.py`):**

1.  **Load Data:**
    * Load `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv` into DataFrames.
    * Ensure `date_index_name` is set as the index for all loaded DataFrames.

2.  **Initial Imputation (on entire DataFrames, including 'Close'):**
    * Identify columns with missing values (if any were not fully addressed by EDA).
    * Apply an initial imputation strategy (e.g., forward-fill, backward-fill, or mean/median for non-time-sensitive features) to the *entire* `train`, `val`, and `test` DataFrames.
    * **Fit any imputer ONLY on the training data**, then transform all three datasets (`train`, `val`, `test`).
    * **Save fitted initial imputer(s)** (e.g., `initial_imputer.pkl` if using scikit-learn imputers).

3.  **Dynamic Market Signal Generation:**
    * **Analyze the embedded `{preliminary_analysis_report_content}` and `{eda_result_summary_content}` to identify potential relationships, trends, or characteristics (e.g., volatility, volume patterns, price movements) that could be good indicators.**
    * **Propose and justify (in `feature_engineering_report_content`) at least 3-5 relevant market signals/features.** Examples could include:
        * **Lagged features:** 'Close' price lagged by 1, 5, 20 days.
        * **Moving Averages (SMA, EMA):** Short-term (e.g., 5-day, 10-day) and long-term (e.g., 20-day, 50-day) moving averages of 'Close' price.
        * **Volatility measures:** Rolling standard deviation of 'Close' price.
        * **Daily Returns:** Percentage change in 'Close' price.
        * **OHLCV-derived features:** High-Low difference, Open-Close difference.
        * **Volume-related features:** Lagged Volume, Volume Rate of Change.
    * **Create these new features** and append them to the DataFrames. Clearly document the chosen features and their parameters in the `feature_engineering_report_content`.

4.  **Post-Signal NaN Handling (on full DataFrames, still including target):**
    * New features (especially lagged or rolling ones) will likely introduce NaNs at the beginning of the time series.
    * Handle these new NaNs. **The recommended strategy is to drop rows with NaNs only if they occur at the very beginning of the series due to feature creation.** This ensures all X and y values align chronologically. If NaNs appear elsewhere, investigate and document for report (may require more sophisticated imputation if not from feature creation).
    * Ensure this dropping is applied consistently across all three datasets (`train`, `val`, `test`) to maintain aligned time periods.

5.  **Separate Target ('Close') into y_sets, remainder is X_sets:**
    * After all feature creation and associated NaN handling, separate the `target_column_name` ('Close') from the features for `train`, `val`, and `test` datasets.
    * Result: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`. Preserve `date_index_name` in all X and y DataFrames.

6.  **Final Imputation (X features only):**
    * Re-check for any remaining NaNs *only* in the `X` feature sets (should primarily be from categorical encoding if not handled, or other edge cases).
    * Apply a final imputation strategy (e.g., median imputation for numerical features).
    * **Fit any imputer ONLY on `X_train`**, then transform `X_train`, `X_val`, `X_test`.
    * **Save fitted final imputer(s)** (e.g., `final_imputer.pkl`).

7.  **Feature Scaling (Numerical X features):**
    * Identify all numerical features in `X_train`.
    * Apply a scaling technique (e.g., `StandardScaler` or `MinMaxScaler`).
    * **Fit the scaler ONLY on `X_train`**, then transform `X_train`, `X_val`, `X_test`.
    * **Save the fitted scaler** (e.g., `scaler.pkl`).

8.  **Categorical Encoding (Categorical X features):**
    * Identify any categorical features (if any were present or created, e.g., 'Category' from dummy data).
    * Apply one-hot encoding or another suitable encoding scheme.
    * **Fit the encoder ONLY on `X_train`**, then transform `X_train`, `X_val`, `X_test`.
    * **Save the fitted encoder** (e.g., `one_hot_encoder.pkl`).
    * Combine the processed numerical and encoded categorical features into the final `X_train`, `X_val`, `X_test` DataFrames, ensuring `date_index_name` is preserved.

9.  **Prepare Report Content:**
    * Compile `feature_engineering_report_content` string detailing:
        * Dynamic choices made for feature generation (justification for each signal).
        * Parameters used for each signal (e.g., window sizes for MAs).
        * Detailed steps for initial imputation, post-signal NaN handling, final imputation, scaling, and encoding.
        * Paths to all saved transformers.
        * Final list of features in the `X` DataFrames.
        * Shapes of `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.

**Python Script Outputs (Files to be created):**
1.  All specified transformer `.pkl` files (e.g., `initial_imputer.pkl`, `final_imputer.pkl`, `scaler.pkl`, `one_hot_encoder.pkl`).
2.  `x_train.csv`, `y_train.csv`, `x_val.csv`, `y_val.csv`, `x_test.csv`, `y_test.csv` (all with `index=True` for `date_index_name`).
3.  `feature_engineering_report.txt`: Text file containing the detailed `feature_engineering_report_content` string.

**Python Script Output Validation (Logic to be included at the end of `feature.py`):**
1.  Verify all `.pkl` and `.csv` files exist.
2.  Load `x_train.csv`, `x_val.csv`, `x_test.csv` into temporary DataFrames.
3.  Check:
    * `date_index_name` is the index for all `x_*.csv` DataFrames.
    * No NaNs in `x_train`, `x_val`, `x_test` (after final imputation).
    * Shapes align as expected.
4.  Load `y_train.csv`, `y_val.csv`, `y_test.csv` into temporary DataFrames.
5.  Check:
    * Shapes align with their respective `X` DataFrames.
    * `date_index_name` is the index for all `y_*.csv` DataFrames.
6.  Attempt to load each `.pkl` file to ensure they are valid (e.g., `joblib.load`).
7.  Print a status message: "Feature engineering outputs validated successfully." or appropriate error messages.

**Preliminary Analysis Report Content (Embedded):**

{preliminary_analysis_report_content}


**EDA Result Summary Content (Embedded):**

{eda_result_summary_content}

"""

model_training_agent_prompt = """
**Goal:**
Generate a Python script (`model_training.py`) to select, train, and tune a regression model.
This script will use feature-engineered datasets and insights from embedded reports.
It must perform hyperparameter tuning using `TimeSeriesSplit`, save the final model,
and report validation metrics.

**Python Script Inputs (for `model_training.py`):**
* `x_train_path`: String, path to `x_train.csv`.
* `y_train_path`: String, path to `y_train.csv`.
* `x_val_path`: String, path to `x_val.csv`.
* `y_val_path`: String, path to `y_val.csv`.
* `preliminary_analysis_report_content`: String, embedded text from Preliminary Analysis Report.
* `eda_result_summary_content`: String, embedded text from EDA Result Summary.
* `feature_engineering_report_content`: String, embedded text from Feature Engineering Report.
* `date_index_name`: String, 'numeric_date_idx'.

**Python Script Instructions (for `model_training.py`):**

1.  **Load Data:**
    * Load `x_train_path`, `y_train_path`, `x_val_path`, `y_val_path` into pandas DataFrames.
    * Ensure `date_index_name` is set as the index for all loaded DataFrames.
    * For `y_train` and `y_val`, convert them to Series if loaded as DataFrames with a single column.

2.  **Dynamic Model Selection:**
    * Analyze the embedded reports, especially `{feature_engineering_report_content}` for feature characteristics (e.g., number of features, linearity, potential interactions).
    * Based on these insights and common practices for time series regression with structured data, **propose 1-2 suitable regression model types**. Justify your choice(s) briefly in the `model_training_report.txt`.
        * Good candidates often include: `XGBoostRegressor`, `LGBMRegressor`, `RandomForestRegressor`.
        * Consider the complexity of the features and the need for capturing non-linear relationships.

3.  **Hyperparameter Tuning:**
    * For each chosen model:
        * Define a reasonable hyperparameter search space. This should be a dictionary for `GridSearchCV` or `RandomizedSearchCV`.
        * **Instantiate `TimeSeriesSplit` from `sklearn.model_selection` as the `cv` strategy.** Choose `n_splits` appropriately for the size of your training data (e.g., 5 or more).
        * **Use `GridSearchCV` or `RandomizedSearchCV`** to perform hyperparameter tuning.
        * Set `scoring='neg_root_mean_squared_error'` for regression, as `GridSearchCV` maximizes scores. The negative sign makes RMSE positive for maximization.
        * Fit the `GridSearchCV`/`RandomizedSearchCV` object using `X_train` and `y_train`.
        * Log the best hyperparameters found to the `model_training_report.txt`.

4.  **Train Final Model:**
    * Instantiate the chosen model type with the best hyperparameters found during tuning.
    * Train this final model on the entire `X_train` and `y_train` datasets.

5.  **Evaluate on Validation Set:**
    * Make predictions on `x_val` using the trained model.
    * Calculate and report the following regression metrics against `y_val`:
        * **Root Mean Squared Error (RMSE)**
        * **Mean Absolute Error (MAE)**
        * **R-squared (R²)**
    * Include these metrics in the `model_training_report.txt`.

6.  **Prepare Report Content:**
    * Compile the `model_training_report_content` string detailing:
        * Justification for the chosen model(s).
        * The hyperparameter search space defined.
        * The best hyperparameters found during tuning.
        * The validation metrics (RMSE, MAE, R²).
        * The path where the final trained model is saved.

**Python Script Outputs (Files to be created):**
1.  `trained_model.pkl`: Saved final trained model (using `joblib.dump`).
2.  `model_training_report.txt`: Text file containing the report content string from step 6 of "Python Script Instructions".

**Python Script Output Validation (Logic to be included at the end of `model_training.py`):**
1.  Verify that `trained_model.pkl` and `model_training_report.txt` were created.
2.  Attempt to load `trained_model.pkl` back (using `joblib.load`) to ensure it's a valid, loadable model object.
3.  Print a status message: "Model training outputs validated successfully." or appropriate error messages if checks fail.

**Preliminary Analysis Report Content (Embedded):**

{preliminary_analysis_report_content}


**EDA Result Summary Content (Embedded):**

{eda_result_summary_content}


**Feature Engineering Report Content (Embedded):**

{feature_engineering_report_content}

"""


evaluation_agent_prompt = """
**Goal:**
Generate a Python script (`model_evaluation.py`) to perform a comprehensive evaluation of the trained regression model on the test set.

**Overall Objective:**
Load the trained model and the test datasets (`x_test.csv`, `y_test.csv`), make predictions, calculate and report key regression metrics, and generate visualizations to assess model performance. A detailed `model_evaluation_report_content` string (also saved as `model_evaluation_report.txt`) must be produced.

**Python Script Inputs (for `model_evaluation.py`):**
* `trained_model_path`: String, path to `trained_model.pkl`.
* `x_test_path`: String, path to `x_test.csv`.
* `y_test_path`: String, path to `y_test.csv`.
* `preliminary_analysis_report_content`: String, embedded text from Preliminary Analysis Report.
* `eda_result_summary_content`: String, embedded text from EDA Result Summary.
* `feature_engineering_report_content`: String, embedded text from Feature Engineering Report.
* `model_training_report_content`: String, embedded text from Model Training Report.
* `date_index_name`: String, 'numeric_date_idx'.
* `target_column_name`: String, 'Close'.

**Python Script Instructions (for `model_evaluation.py`):**

1.  **Load Data and Model:**
    * Load `x_test_path` and `y_test_path` into pandas DataFrames, ensuring `date_index_name` is set as the index.
    * Load the `trained_model.pkl` using `joblib`.
    * For `y_test`, convert to a Series if loaded as a DataFrame with a single column.

2.  **Make Predictions:**
    * Use the loaded model to make predictions on `X_test`.

3.  **Calculate Evaluation Metrics:**
    * Calculate the following regression metrics for the predictions against `y_test`:
        * **Root Mean Squared Error (RMSE)**
        * **Mean Absolute Error (MAE)**
        * **R-squared (R²)**
        * **Mean Absolute Percentage Error (MAPE)** - (You may need to define a custom function for this if not directly available in sklearn, handling division by zero for actual values of 0).

4.  **Generate Visualizations:**
    * **Time Series Plot of Actual vs. Predicted:** Plot `y_test` (actual values) and the model's predictions over time. This is crucial for time series data to visually inspect how well the model tracks trends and volatility.
    * **Residuals Plot:** Plot the residuals (actual - predicted) against time or against predicted values. This helps in identifying patterns in errors (e.g., heteroscedasticity, autocorrelation).
    * **Actual vs. Predicted Scatter Plot:** A scatter plot of actual `y_test` values against predicted values. Ideally, points should cluster around a 45-degree line.

5.  **Prepare Report Content:**
    * Compile `model_evaluation_report_content` string detailing:
        * All calculated evaluation metrics (RMSE, MAE, R², MAPE).
        * Interpretation of the metrics.
        * Observations from the generated visualizations (e.g., model's ability to capture trends, presence of systematic errors).
        * A summary of the model's overall performance and any identified strengths or weaknesses.
        * Suggestions for future improvements or further analysis based on the evaluation.

**Python Script Outputs (Files to be created):**
1.  `model_evaluation_report.txt`: Text file containing the detailed `model_evaluation_report_content` string.
2.  `actual_vs_predicted_plot.png`: Saved plot of actual vs. predicted values over time.
3.  `residuals_plot.png`: Saved plot of residuals.
4.  `actual_vs_predicted_scatter.png`: Saved scatter plot of actual vs. predicted values.

**Python Script Output Validation (Logic to be included at the end of `model_evaluation.py`):**
1.  Verify that `model_evaluation_report.txt`, `actual_vs_predicted_plot.png`, `residuals_plot.png`, and `actual_vs_predicted_scatter.png` were created.
2.  Print a status message: "Model evaluation outputs validated successfully." or appropriate error messages if checks fail.

**Agent's Final Return Values (for LangGraph state):**
* The string content of `model_evaluation_report.txt`.
"""
