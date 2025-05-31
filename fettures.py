**Goal:**
Generate a Python script named `preliminary_script.py`. This script will:
1. Load raw stock data from the provided `raw_data_path`.
2. Perform initial data inspection: print shape, `.info()`, `.head()`, `.describe(include='all')`, initial missing value counts (`.isnull().sum()`), and unique value counts per column (`.nunique()`).
3. Handle dates: Identify the primary 'Date' column, parse it to pandas datetime objects, sort the DataFrame chronologically by this 'Date'. Create a `numeric_date_idx` column (e.g., Unix timestamp seconds from 'Date'), set `numeric_date_idx` as the DataFrame's index, and ensure the original 'Date' column (as datetime objects) is retained.
4. Save all printed inspection outputs and date handling details into a text file named `preliminary_analysis_report.txt`.
5. Save the fully processed DataFrame (with 'Date' column and `numeric_date_idx` as index) to `initially_processed_data.csv` using `index=True`.
6. Validate outputs: The script must check that both `initially_processed_data.csv` and `preliminary_analysis_report.txt` were created. It should then load `initially_processed_data.csv`, verify its shape, confirm `numeric_date_idx` is its index, and that the 'Date' column exists. Print a validation status message.

**Inputs (Provided to Agent):**
* `raw_data_path`: (e.g., "path/to/your/raw_stock_data.csv")



**Goal:**
You are an expert AI assistant. Your task is to generate a detailed and actionable **prompt string**. This prompt string will be given to a subsequent `EDAAgent` to guide its generation of an `eda_script.py`. Base your generated prompt *exclusively* on the findings within the provided `{preliminary_analysis_report_content}`.

**Input (Provided to you, EDAPromptGeneratorAgent):**
* `{preliminary_analysis_report_content}`: (Full text content from the `PreliminaryAnalysisAgent`'s `preliminary_analysis_report.txt`)

**Output (The prompt string you must generate for the EDAAgent):**
The prompt string you generate must instruct the `EDAAgent` to create an `eda_script.py` that achieves the following:
* **Overall Objective:** Perform detailed EDA and data cleaning on `initially_processed_data.csv`. The script MUST leverage the specific insights from the embedded `{preliminary_analysis_report_content}` (which you will place into the prompt you are generating). After cleaning and EDA, it must chronologically split the data (e.g., using 0.7 train, 0.15 val, 0.15 test ratios) into `Cleaned_Train.csv`, `Cleaned_Val.csv`, and `Cleaned_Test.csv`, ensuring the `numeric_date_idx` is preserved as the index and all original columns (including 'Close' and OHLCV) are retained in these splits. A detailed `eda_result_summary.txt` must also be produced.
* **Specific Instructions for the `EDAAgent`'s script (to be detailed in your generated prompt):**
    1.  Load `initially_processed_data.csv` (confirming `numeric_date_idx` as index).
    2.  Perform targeted data cleaning (e.g., type corrections, handling specific issues noted in the embedded `{preliminary_analysis_report_content}`).
    3.  Conduct in-depth EDA on the conceptual training portion of the data: generate and save key visualizations (or provide detailed textual descriptions for the report), analyze distributions, correlations, and time-series patterns for key columns like 'Close' and 'Volume'.
    4.  Chronologically split the full, cleaned DataFrame into train, validation, and test sets.
    5.  Save `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv` (all with `index=True` to preserve `numeric_date_idx`).
    6.  Save a comprehensive `eda_result_summary.txt` detailing all EDA findings, cleaning actions, visualization summaries, and split details (shapes, date ranges).
    7.  Ensure the `eda_script.py` validates its own outputs: check all specified CSV and TXT files are created; load back CSVs to verify index, shapes, and presence of key columns like 'Close'. Print validation status.





  **Goal:**
Generate a Python script named `feature.py` for dynamic and comprehensive feature engineering. This script will use the `Cleaned_*.csv` datasets and insights from the embedded `{preliminary_analysis_report_content}` and `{eda_result_summary_content}`. It must dynamically propose, justify, and create relevant market signals with appropriate parameters based on this analysis.

The script must strictly follow this operational sequence:
1. Load `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv` (these contain 'Close', OHLCV, and have `numeric_date_idx` as index).
2. Perform initial missing value imputation on these entire DataFrames (fit imputers for numerical e.g., 'median'/'ffill'; and categorical e.g., 'most_frequent' *ONLY* on `train_df`. Save these as `initial_numerical_imputer.pkl` and `initial_categorical_imputer.pkl`. Transform `train_df`, `val_df`, `test_df`).
3. Dynamically generate market signals (e.g., MAs, Lags, RSI, MACD, Volatility) using OHLCV columns (including 'Close') from the imputed DataFrames from step 2. Append these new features. Log created signals and their parameters for the report.
4. Handle NaNs introduced by new signals: Drop rows containing these new NaNs from the DataFrames (which still include the target 'Close'). This must be done consistently across train, val, and test sets to maintain data alignment for target separation.
5. Separate Target: From the NaN-handled DataFrames, separate the `target_column_name` ('Close') into `y_train`, `y_val`, `y_test`. The remaining columns form `X_train`, `X_val`, `X_test`.
6. Perform final missing value imputation *only* on `X_train`, `X_val`, `X_test` feature sets (fit imputers *ONLY* on `X_train`. Save as `final_numerical_feature_imputer.pkl`, `final_categorical_feature_imputer.pkl`. Transform all X sets). This is a safety net.
7. Perform feature scaling on numerical features in X sets (fit `StandardScaler` *ONLY* on `X_train`. Save as `scaler.pkl`. Transform all X sets).
8. Perform categorical feature encoding on categorical features in X sets (fit `OneHotEncoder` with `handle_unknown='ignore'` *ONLY* on `X_train`. Save as `one_hot_encoder.pkl`. Transform all X sets).
9. Combine all processed X features, ensuring `numeric_date_idx` is preserved as the index.
10. Save all specified output files and validate them within the script.

**Inputs (Agent will receive):**
* `cleaned_train_path`: Path to `Cleaned_Train.csv`.
* `cleaned_val_path`: Path to `Cleaned_Val.csv`.
* `cleaned_test_path`: Path to `Cleaned_Test.csv`.
* `{preliminary_analysis_report_content}`: Embedded text content.
* `{eda_result_summary_content}`: Embedded text content.
* `target_column_name`: 'Close'.
* `date_index_name`: 'numeric_date_idx'.

**Outputs & Validation (to be implemented in `feature.py`):**
* **Fitted Transformers (saved with `joblib`):** `initial_numerical_imputer.pkl`, `initial_categorical_imputer.pkl`, `final_numerical_feature_imputer.pkl`, `final_categorical_feature_imputer.pkl`, `scaler.pkl`, `one_hot_encoder.pkl`.
* **Processed Datasets (CSVs with `date_index_name` as index, use `index=True`):** `x_train.csv`, `y_train.csv`, `x_val.csv`, `y_val.csv`, `x_test.csv`, `y_test.csv`.
* **Feature Engineering Report (`feature_engineering_report.txt`):** Detail dynamic feature choices (signals, parameters, justifications based on EDA/data), all processing steps, names of saved transformers, final feature list in `x_train.csv`, and final data shapes.
* **Validate Outputs (in script):** Check all specified `.pkl` and `.csv` files exist. Load `x_*.csv` files, verify index, check for NaNs (should be none/minimal), check shapes. Load `y_*.csv` files, verify shape alignment with corresponding X files. Print validation status.




**Goal:**
Generate a Python script named `model_training.py`. This script will select an appropriate regression model, train it, and tune its hyperparameters using the provided feature-engineered datasets and insights from embedded reports (`{preliminary_analysis_report_content}`, `{eda_result_summary_content}`, and crucially `{feature_engineering_report_content}`). It must perform hyperparameter tuning using `TimeSeriesSplit` for cross-validation with the validation set approach. The final trained model and a report detailing its training and validation metrics must be saved.

**Inputs (Agent will receive):**
* `x_train_path`: Path to `x_train.csv`.
* `y_train_path`: Path to `y_train.csv`.
* `x_val_path`: Path to `x_val.csv`.
* `y_val_path`: Path to `y_val.csv`.
* `{preliminary_analysis_report_content}`: Embedded text content.
* `{eda_result_summary_content}`: Embedded text content.
* `{feature_engineering_report_content}`: Embedded text content.
* `date_index_name`: 'numeric_date_idx'.

**Script (`model_training.py`) Instructions:**

1.  **Load Data:** Load data from `x_train_path`, `y_train_path`, `x_val_path`, `y_val_path`, ensuring `date_index_name` is set as index for X DataFrames and y Series.
2.  **Dynamic Model Selection:**
    * Analyze the embedded `{feature_engineering_report_content}` (especially feature characteristics) and `{eda_result_summary_content}`.
    * Propose one or two suitable regression model types (e.g., XGBoost, LightGBM, RandomForestRegressor) and provide a brief justification for your choice(s).
3.  **Hyperparameter Tuning:**
    * For the chosen model(s), define a relevant hyperparameter search space.
    * Use `GridSearchCV` or `RandomizedSearchCV` from `sklearn.model_selection`.
    * **CRITICAL: For the `cv` parameter, use `TimeSeriesSplit(n_splits=...)` from `sklearn.model_selection`.**
    * Use an appropriate regression scoring metric (e.g., `'neg_root_mean_squared_error'`).
    * Fit the search object using the full `x_train` and `y_train` data. (Note: `TimeSeriesSplit` will handle the internal splitting of this training data for cross-validation).
4.  **Train Final Model:** After identifying the best hyperparameters, instantiate the model with these optimal parameters and train it on the entire `x_train` and `y_train` data.
5.  **Evaluate on Validation Set:** Use the final trained model to make predictions on `x_val`. Calculate and report RMSE, MAE, and R-squared against `y_val`.
6.  **Outputs & Validation (within the script):**
    * **Save Model:** Save the final trained model object to `trained_model.pkl` using `joblib`.
    * **Save Training Report:** Create `model_training_report.txt` detailing: model choice and justification, hyperparameter search space, best hyperparameters found, validation metrics (RMSE, MAE, R²) on `x_val`/`y_val`, and the path to `trained_model.pkl`.
    * **Validate Outputs:** Check that `trained_model.pkl` and `model_training_report.txt` were created. Attempt to load `trained_model.pkl` back to ensure it's valid. Print validation status.
