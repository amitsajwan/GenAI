**Overall Goal:**
Your primary mission is to act as an expert Feature Engineering specialist. Devise and then generate a comprehensive Python script, to be named `feature.py`. This script must intelligently perform feature engineering on the provided cleaned stock market datasets. It should analyze the input data and accompanying reports to dynamically propose, create, and justify relevant domain-specific features (market signals) with appropriate parameters. The script must rigorously follow a specific sequence of operations: initial imputation on all available data (including OHLCV columns for signal generation), creation of these new market signals, meticulous handling of any NaNs introduced by new features (by dropping affected rows to ensure data integrity and alignment), followed by the separation of the target variable. After target separation, the script must apply any necessary final imputation, robust feature scaling, and consistent categorical feature encoding to the resulting feature sets (X_train, X_val, X_test). All resulting processed datasets and all fitted data transformers (imputers, scalers, encoders) MUST be saved to disk. Finally, a detailed textual report summarizing all actions, decisions, created features, saved files, and final data characteristics must be generated. Crucially, ensure the date index present in the input files is meticulously preserved in all output feature DataFrames.

**Input Context (to be found in the current directory):**
1.  `Cleaned_Train.csv`: Primary training data.
2.  `Cleaned_Val.csv`: Validation data.
3.  `Cleaned_Test.csv`: Test data.
    *(These CSVs are expected to contain 'Open', 'High', 'Low', 'Close', 'Volume' (OHLCV) columns and other existing features, along with a date-based index, assumed to be 'numeric_date_idx' or otherwise inferable and specified in reports.)*
4.  `eda_result_summary.txt`: Leverage this for insights into data characteristics, distributions, correlations, and potential feature ideas.
5.  `preliminary_analysis_report.txt`: Use this for further context on the raw data and initial findings.

**Key Variables:**
* **Target Variable Name:** 'Close'
* **Date Index Name:** Assume 'numeric_date_idx' (this should be the actual index of the input DataFrames). This index MUST be preserved.

**Mandatory Operational Sequence for `feature.py`:**

1.  **Data Ingestion:**
    * Load `Cleaned_Train.csv`, `Cleaned_Val.csv`, `Cleaned_Test.csv` into pandas DataFrames (`train_df`, `val_df`, `test_df`), ensuring the specified date index is correctly loaded and set. These initial DataFrames contain all columns, including OHLCV and the future target.

2.  **Initial Universal Imputation:**
    * **Objective:** Prepare all columns, especially OHLCV, for reliable technical indicator calculation by handling existing missing values.
    * **Process:**
        * Distinguish numerical and categorical columns in `train_df`.
        * For numerical columns: Select an appropriate imputation strategy (e.g., 'median', 'mean', or time-series appropriate like 'ffill'/'bfill' – justify your choice based on data/reports if possible). Fit the imputer *ONLY* on `train_df`. Save this imputer as `initial_numerical_imputer.pkl`. Apply it to transform `train_df`, `val_df`, and `test_df`.
        * For categorical columns: Select an appropriate strategy (e.g., 'most_frequent', 'constant'). Fit the imputer *ONLY* on `train_df`. Save as `initial_categorical_imputer.pkl`. Apply it to transform `train_df`, `val_df`, and `test_df`.
    * Designate these processed DataFrames (e.g., `train_df_imputed1`, etc.).

3.  **Dynamic Market Signal Generation (Core Reasoning Task):**
    * **Objective:** Enhance the feature set with predictive market signals.
    * **Process:**
        * **Analyze & Propose:** Scrutinize `eda_result_summary.txt`, `preliminary_analysis_report.txt`, and the characteristics of `train_df_imputed1`. Based on this analysis, propose a set of relevant market signals (e.g., Moving Averages, Lag Features, Momentum Indicators like RSI/MACD, Volatility Measures like Bollinger Bands/ATR, etc.).
        * **Parameterize Intelligently:** For each chosen signal, determine and justify appropriate parameters (e.g., window sizes, lag periods). Your choices should be data-driven (e.g., based on cycles or autocorrelations noted in EDA) or based on well-established financial analysis practices.
        * **Implement Creation:** Generate Python code to calculate these signals using the OHLCV columns (and potentially others) from `train_df_imputed1`, `val_df_imputed1`, `test_df_imputed1`. Append these new features to these DataFrames. You may use pandas or consider appropriate libraries like `TA-Lib` or `pandas_ta` (if so, list them as dependencies in your report/script comments).
    * Designate these enriched DataFrames (e.g., `train_df_with_signals`, etc.).
    * **Log all created signals and their parameters meticulously for the final report.**

4.  **Post-Signal NaN Handling (Critical for Alignment):**
    * **Objective:** Ensure data integrity after signal generation.
    * **Process:** Lag and rolling window calculations will introduce NaNs. **Drop all rows containing these newly introduced NaNs** from `train_df_with_signals`, `val_df_with_signals`, and `test_df_with_signals`. This must be done consistently across all three datasets to maintain alignment for later target separation.
    * Designate these complete-case DataFrames (e.g., `train_df_final_structure`, etc.).

5.  **Target Variable Separation:**
    * From `train_df_final_structure`, `val_df_final_structure`, `test_df_final_structure`, separate the 'Close' column into `y_train`, `y_val`, `y_test`.
    * The remaining columns form the feature sets `X_train`, `X_val`, `X_test`.

6.  **Final Feature Set Imputation (Safety Net for X features):**
    * **Objective:** Handle any extremely rare residual NaNs in the `X_train`, `X_val`, `X_test` feature sets.
    * **Process:** Similar to Step 2, but applied *only* to `X_train`, `X_val`, `X_test`. Fit imputers *ONLY* on `X_train`. Save them as `final_numerical_feature_imputer.pkl` and `final_categorical_feature_imputer.pkl`. Transform all X sets.
    * Designate these (e.g., `X_train_imputed2`, etc.).

7.  **Feature Scaling (Numerical X features):**
    * **Objective:** Standardize numerical feature scales.
    * **Process:** Apply to numerical columns of `X_train_imputed2` (and corresponding val/test sets). Initialize a `StandardScaler`. Fit *ONLY* on `X_train_imputed2`. Save as `scaler.pkl`. Transform all X sets.

8.  **Categorical Feature Encoding (Categorical X features):**
    * **Objective:** Convert categorical features into a model-usable numerical format.
    * **Process:** Apply to categorical columns of `X_train_imputed2` (or the version after scaling, as appropriate). Initialize `OneHotEncoder` (use `handle_unknown='ignore'`, `sparse_output=False`). Fit *ONLY* on `X_train_imputed2`. Save as `one_hot_encoder.pkl`. Transform all X sets.
    * Combine all processed numerical (scaled) and categorical (encoded) features to form the final `X_train_final`, `X_val_final`, `X_test_final`. Ensure the date index is meticulously preserved.

**Outputs (Script `feature.py` must create these):**

1.  **Fitted Transformer Files (using `joblib`):**
    * `initial_numerical_imputer.pkl`
    * `initial_categorical_imputer.pkl`
    * `final_numerical_feature_imputer.pkl`
    * `final_categorical_feature_imputer.pkl`
    * `scaler.pkl`
    * `one_hot_encoder.pkl`

2.  **Processed Datasets (CSVs with preserved index):**
    * `x_train.csv` (from `X_train_final`)
    * `y_train.csv`
    * `x_val.csv` (from `X_val_final`)
    * `y_val.csv`
    * `x_test.csv` (from `X_test_final`)
    * `y_test.csv`

3.  **Comprehensive Textual Report (`feature_engineering_report.txt`):**
    * This report should be human-readable and detail:
        * Timestamp of execution.
        * A summary of the entire process.
        * **Key decisions made for feature creation:** Which market signals were generated, what parameters were used (e.g., window sizes, lag periods), and a brief justification for these choices, especially if informed by the EDA reports.
        * **Imputation strategies** used at each stage (initial and final) and for which types of columns.
        * How NaNs introduced by signal generation were handled (e.g., "Dropped X rows...").
        * **Scaling and encoding methods** applied.
        * **List of all saved transformer files.**
        * **Final list of feature names** present in `x_train.csv`.
        * **Final shapes** (rows, columns) of `x_train.csv`, `x_val.csv`, and `x_test.csv`.
        * Any significant observations, assumptions made, or potential limitations of the feature engineering process.

**Final Script (`feature.py`) Considerations:**
* Must use `pandas`, `numpy`, `sklearn`, `joblib`.
* All file paths should be relative to the script's execution directory.
* Include informative print statements for major steps.
* Strive for robust code that can handle minor variations or edge cases (e.g., a dataset with no categorical features after initial cleaning).

This prompt gives GPT-4o the operational skeleton and constraints but allows its reasoning to fill in the "intelligent" parts, particularly in step 3 (Dynamic Market Signal Generation) and in how it justifies its choices in the final report.
