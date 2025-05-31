**Agent Goal:**
Generate a Python script that performs initial loading, inspection, and date standardization for raw stock data, then saves a report and the processed DataFrame.

**Agent Inputs:**
* `raw_data_path`: String, path to the raw stock data CSV file.

**Agent Output Script Name:** `preliminary_script.py`

**Python Script Instructions (for `preliminary_script.py`):**

1.  **Load & Inspect Data:**
    * Load the CSV from `raw_data_path` into a pandas DataFrame.
    * Collect for reporting: DataFrame shape, `.info()` output, `.head(5)` output, `.describe(include='all')` output, initial missing value counts per column (`.isnull().sum()`), and unique value counts per column (`.nunique()`).
2.  **Date Handling & Indexing:**
    * Identify and parse the primary 'Date' column to pandas datetime objects. Handle parsing issues gracefully and note them for the report.
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

**Agent's Final Return Values (for LangGraph state):**
* Path to `initially_processed_data.csv`.
* The string content of `preliminary_analysis_report.txt`.













**Agent Goal:**
Generate a detailed and actionable **prompt string**. This prompt string will be used by a subsequent `EDAAgent` to guide its Python script generation (`eda_script.py`). The generated prompt must be based *exclusively* on the findings within the provided `{preliminary_analysis_report_content}`.

**Agent Inputs:**
* `{preliminary_analysis_report_content}`: String, full text content from the `PreliminaryAnalysisAgent`'s `preliminary_analysis_report.txt`.

**Agent Output:**
* A single, well-structured **prompt string** to be used by the `EDAAgent`.

**Key Requirements for the Generated Prompt String (for the `EDAAgent`):**
The prompt string you generate must instruct the `EDAAgent` that its `eda_script.py` should achieve the following:
* **Overall Objective:** Perform detailed EDA and cleaning on `initially_processed_data.csv` (path to be provided to `EDAAgent`). The script MUST leverage specific insights from the embedded `{preliminary_analysis_report_content}` (which you will place into the prompt you are generating for the EDAAgent). After cleaning and EDA, it must chronologically split the data (e.g., 0.7 train, 0.15 val, 0.15 test ratios) into `Cleaned_Train.csv`, `Cleaned_Val.csv`, and `Cleaned_Test.csv` (preserving `numeric_date_idx` as index and ensuring 'Close' and other OHLCV columns are included). A detailed `eda_result_summary_content` string (also saved as `eda_result_summary.txt`) must be produced.
* **Script Inputs for `eda_script.py`:**
    * Path to `initially_processed_data.csv`.
    * The embedded `{preliminary_analysis_report_content}`.
    * Split Ratios.
* **Script Instructions for `eda_script.py`:**
    1.  Load `initially_processed_data.csv` (confirming `numeric_date_idx` as index).
    2.  Perform targeted data cleaning based *specifically* on issues noted in embedded `{preliminary_analysis_report_content}`.
    3.  Conduct in-depth EDA on the conceptual training portion (generate/save key visualizations or provide detailed textual descriptions for report; analyze distributions, correlations, time-series patterns for 'Close', 'Volume').
    4.  Chronologically split the full, cleaned DataFrame. Ensure 'Close' and OHLCV columns are present.
* **Script Outputs for `eda_script.py`:**
    * `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv` (all with `index=True` for `numeric_date_idx`).
    * `eda_result_summary.txt` (containing the detailed `eda_result_summary_content` string).
* **Script Output Validation for `eda_script.py`:**
    * Check all specified CSV and TXT files are created.
    * Load each `Cleaned_*.csv`; verify `numeric_date_idx` is index, shapes align with ratios, 'Close' and other key OHLCV columns exist. Print validation status.
* **Agent's Final Return Values (for `EDAAgent` node):** Paths to cleaned CSVs and the `eda_result_summary_content` string.





**Agent Goal:**
Generate a detailed and actionable **prompt string**. This prompt string will be used by a subsequent `FeatureEngineeringAgent` to guide its Python script generation (`feature.py`). The generated prompt must be based *exclusively* on the findings within the provided `{preliminary_analysis_report_content}` and `{eda_result_summary_content}`.

**Agent Inputs:**
* `{preliminary_analysis_report_content}`: String, full text content from `PreliminaryAnalysisAgent`.
* `{eda_result_summary_content}`: String, full text content from `EDAAgent`.

**Agent Output:**
* A single, well-structured **prompt string** to be used by the `FeatureEngineeringAgent`.

**Key Requirements for the Generated Prompt String (for the `FeatureEngineeringAgent`):**
The prompt string you generate must instruct the `FeatureEngineeringAgent` that its `feature.py` script should achieve the following:
* **Overall Objective:** Perform dynamic and comprehensive feature engineering using `Cleaned_*.csv` data (paths to be provided to `FeatureEngineeringAgent`). The script MUST leverage insights from embedded `{preliminary_analysis_report_content}` and `{eda_result_summary_content}` (which you will place in the prompt you are generating) to dynamically propose, justify, and create relevant market signals. It must follow a strict order: 1. Initial imputation (on full DataFrames, including 'Close'). 2. New market signal creation (using 'Close' and other OHLCV from full, imputed DataFrames). 3. Handle NaNs from new signals (row drops on full DataFrames, ensuring y-alignment later). 4. THEN, separate target ('Close'). 5. Finally, apply final imputation/scaling/encoding to X features (fit ONLY on train, transform all, save transformers). All outputs (processed X/Y datasets, fitted transformers) must be saved, along with a detailed `feature_engineering_report_content` string (also saved as `feature_engineering_report.txt`). `numeric_date_idx` MUST be preserved.
* **Script Inputs for `feature.py`:**
    * Paths to `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv`.
    * Embedded `{preliminary_analysis_report_content}` and `{eda_result_summary_content}`.
    * `target_column_name`: 'Close'.
    * `date_index_name`: 'numeric_date_idx'.
* **Script Key Operational Sequence for `feature.py` (abbreviated - you will detail this based on our prior discussions):**
    1.  Load Data.
    2.  Initial Imputation (on entire DFs, fit on train, save `initial_*.pkl`, transform all).
    3.  Dynamic Market Signal Generation (analyze reports, propose/justify/create/parameterize signals from OHLCV including 'Close', append, log for report).
    4.  Post-Signal NaN Handling (drop rows from DFs still including target).
    5.  Separate Target ('Close') into y_sets, remainder is X_sets.
    6.  Final Imputation (X features only, fit on X_train, save `final_*.pkl`, transform X_sets).
    7.  Feature Scaling (Numerical X, fit on X_train, save `scaler.pkl`, transform X_sets).
    8.  Categorical Encoding (Categorical X, fit on X_train, save `one_hot_encoder.pkl`, transform X_sets). Combine X features, preserve index.
* **Script Outputs for `feature.py`:**
    * All specified transformer `.pkl` files.
    * `x_train.csv`, `y_train.csv`, etc. (all with `index=True` for `date_index_name`).
    * `feature_engineering_report.txt` (containing the detailed `feature_engineering_report_content` string: dynamic choices, params, all steps, transformers, final feature list, shapes).
* **Script Output Validation for `feature.py`:**
    * Check all `.pkl` & `.csv` files exist. Load `x_*.csv`, verify index, no NaNs, shapes. Load `y_*.csv`, verify shape alignment. Print validation status.
* **Agent's Final Return Values (for `FeatureEngineeringAgent` node):** Paths to X/Y CSVs, transformer paths, and the `feature_engineering_report_content` string.





**Agent Goal:**
Generate a Python script (`model_training.py`) to select, train, and tune a regression model. This script will use feature-engineered datasets and insights from embedded reports. It must perform hyperparameter tuning using `TimeSeriesSplit`, save the final model, and report validation metrics.

**Agent Inputs:**
* `x_train_path`: Path to `x_train.csv`.
* `y_train_path`: Path to `y_train.csv`.
* `x_val_path`: Path to `x_val.csv`.
* `y_val_path`: Path to `y_val.csv`.
* `{preliminary_analysis_report_content}`: Embedded text.
* `{eda_result_summary_content}`: Embedded text.
* `{feature_engineering_report_content}`: Embedded text.
* `date_index_name`: 'numeric_date_idx'.

**Agent Output Script Name:** `model_training.py`

**Python Script Instructions (for `model_training.py`):**

1.  **Load Data:** Load from input paths, setting `date_index_name` as index.
2.  **Dynamic Model Selection:**
    * Analyze embedded reports (especially `{feature_engineering_report_content}` for feature characteristics).
    * Propose 1-2 suitable regression model types (e.g., XGBoost, LightGBM) with brief justification.
3.  **Hyperparameter Tuning:**
    * For chosen model(s), define search space. Use `GridSearchCV` or `RandomizedSearchCV`.
    * **Use `TimeSeriesSplit` (from `sklearn.model_selection`) as the `cv` strategy.**
    * Use regression scoring (e.g., `'neg_root_mean_squared_error'`). Fit using `x_train` and `y_train`.
4.  **Train Final Model:** With best HPs, train on entire `x_train`, `y_train`.
5.  **Evaluate on Validation Set:** Predict on `x_val` and calculate/report RMSE, MAE, R² against `y_val`.
6.  **Prepare Report Content:** Compile model choice justification, HPs, validation metrics, and path to model into a single string.

**Python Script Outputs (Files to be created):**
1.  `trained_model.pkl`: Saved final trained model (using `joblib`).
2.  `model_training_report.txt`: Text file containing the report content string from step 6 of "Python Script Instructions".

**Python Script Output Validation (Logic to be included at the end of `model_training.py`):**
1.  Verify `trained_model.pkl` and `model_training_report.txt` were created.
2.  Attempt to load `trained_model.pkl` back to ensure it's valid.
3.  Print a status message: "Model training outputs validated successfully." or appropriate error messages.

**Agent's Final Return Values (for LangGraph state):**
* Path to `trained_model.pkl`.
* The string content of `model_training_report.txt`.






