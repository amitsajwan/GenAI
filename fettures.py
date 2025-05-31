**Goal:**
Generate a Python script named `feature.py` to perform comprehensive feature engineering on the provided cleaned stock market datasets. The script MUST first perform initial imputation on all available data (including OHLCV columns), then create new domain-specific features (market signals), handle any NaNs introduced by these new features by dropping affected rows, then separate the target variable. Finally, it should apply further imputation (if needed on remaining features), feature scaling, and categorical feature encoding consistently across train, validation, and test feature sets (X). All resulting datasets and fitted transformers must be saved. Ensure the date index present in the input files is preserved.

**Input Files (located in the current directory):**
1.  `Cleaned_Train.csv` (training data - expected to contain 'Open', 'High', 'Low', 'Close', 'Volume' (OHLCV) and other features)
2.  `Cleaned_Val.csv` (validation data - similar structure)
3.  `Cleaned_Test.csv` (test data - similar structure)
4.  `eda_result_summary.txt` (for insights, if helpful)
5.  `preliminary_analysis_report.txt` (for insights, if helpful)

**Target Variable:**
The target variable to be separated is named 'Close'.

**Date Index:**
The input CSV files have a date index (assume it's named 'numeric_date_idx' or infer from data if no specific name given in reports). This index MUST be preserved in all output feature DataFrames (`x_train.csv`, `x_val.csv`, `x_test.csv`) and target Series/DataFrames (`y_train.csv`, etc.).

**Core Tasks for `feature.py` (MUST be performed in this specific order):**

1.  **Load Data:**
    * Load `Cleaned_Train.csv`, `Cleaned_Val.csv`, and `Cleaned_Test.csv` into pandas DataFrames: `train_df`, `val_df`, `test_df`. Ensure the date index is correctly set upon loading. These DataFrames contain all features, including the future target 'Close' and other OHLCV columns.

2.  **Initial Missing Value Imputation (on ENTIRE DataFrames):**
    * **Purpose:** To ensure OHLCV columns (like 'Close', 'Volume', etc.) are complete *before* calculating technical indicators that depend on them.
    * **Identify Column Types:** Distinguish between numerical and categorical columns in `train_df`.
    * **Numerical Imputation:**
        * Initialize a `SimpleImputer` (e.g., strategy 'median', or 'ffill' for time-series appropriateness). Use insights from reports if possible.
        * **Fit this numerical imputer *ONLY* on the numerical columns of `train_df`.**
        * **Save as `initial_numerical_imputer.pkl`**.
        * Use this *fitted* imputer to **transform** numerical columns in `train_df`, `val_df`, and `test_df`.
    * **Categorical Imputation:**
        * Initialize a `SimpleImputer` (e.g., strategy 'most_frequent').
        * **Fit this categorical imputer *ONLY* on the categorical columns of `train_df`.**
        * **Save as `initial_categorical_imputer.pkl`**.
        * Use this *fitted* imputer to **transform** categorical columns in `train_df`, `val_df`, and `test_df`.
    * Let the resulting DataFrames be `train_df_imputed1`, `val_df_imputed1`, `test_df_imputed1`.

3.  **Domain-Specific Feature Creation (Market Signals):**
    * Use the `train_df_imputed1`, `val_df_imputed1`, `test_df_imputed1` DataFrames. These DataFrames still contain 'Open', 'High', 'Low', 'Close', 'Volume' and other original (now imputed) features.
    * Generate a set of common technical indicators. You may use pandas for rolling calculations, or libraries like `TA-Lib` or `pandas_ta` if available (note them as dependencies). Consider features like:
        * Moving Averages (e.g., SMA_7, SMA_21, EMA_14) for the 'Close' price.
        * Lag Features (e.g., 'Close_lag_1', 'Volume_lag_1').
        * Rate of Change (ROC).
        * Relative Strength Index (RSI).
        * Moving Average Convergence Divergence (MACD).
        * Bollinger Bands (Upper, Middle, Lower).
        * Volatility (e.g., rolling standard deviation of 'Close' price returns).
    * Choose appropriate window sizes (e.g., 7, 14, 21, 50) or use insights from EDA reports.
    * Append these new signal features to `train_df_imputed1`, `val_df_imputed1`, and `test_df_imputed1`. Let the results be `train_df_with_signals`, `val_df_with_signals`, `test_df_with_signals`.

4.  **Handle NaNs Introduced by New Features (CRITICAL):**
    * Creating lag or rolling window features will introduce NaNs (typically at the beginning of each series).
    * **After generating all new signal features, drop any rows from `train_df_with_signals`, `val_df_with_signals`, and `test_df_with_signals` that now contain NaNs due to these calculations.** This ensures data alignment and completeness before splitting X and y.
    * Let the resulting DataFrames (with fewer rows but complete data) be `train_df_final_structure`, `val_df_final_structure`, `test_df_final_structure`.

5.  **Separate Target Variable:**
    * From `train_df_final_structure`, `val_df_final_structure`, and `test_df_final_structure`:
        * Separate the target variable ('Close') into `y_train`, `y_val`, and `y_test`.
        * The remaining columns (original imputed features + new market signals, excluding 'Close') will form `X_train`, `X_val`, and `X_test`.

6.  **Final Missing Value Imputation (on Feature Sets `X_train`, `X_val`, `X_test`):**
    * **Purpose:** To handle any *very rare* residual NaNs in the *feature columns* of `X_train`, `X_val`, `X_test` that might have appeared from other complex operations or were not perfectly handled by the row drops. This is a safety net.
    * **Identify Column Types:** Distinguish numerical and categorical columns in `X_train`.
    * **Numerical Imputation (for X features):**
        * Initialize a `SimpleImputer` (e.g., 'median').
        * **Fit *ONLY* on numerical columns of `X_train`.** Save as `final_numerical_feature_imputer.pkl`.
        * Transform numerical columns in `X_train`, `X_val`, and `X_test`.
    * **Categorical Imputation (for X features):**
        * Initialize a `SimpleImputer` (e.g., 'most_frequent').
        * **Fit *ONLY* on categorical columns of `X_train`.** Save as `final_categorical_feature_imputer.pkl`.
        * Transform categorical columns in `X_train`, `X_val`, and `X_test`.
    * Let the results be `X_train_imputed2`, `X_val_imputed2`, `X_test_imputed2`.

7.  **Feature Scaling (Numerical Features in `X` sets - CRITICAL FOR CONSISTENCY):**
    * Apply to numerical columns of `X_train_imputed2`, `X_val_imputed2`, `X_test_imputed2`.
    * Initialize a `StandardScaler`.
    * **Fit `StandardScaler` *ONLY* on numerical columns of `X_train_imputed2`.**
    * **Save as `scaler.pkl`**.
    * Use it to **transform** numerical columns in `X_train_imputed2`, `X_val_imputed2`, and `X_test_imputed2`.

8.  **Categorical Feature Encoding (Categorical Features in `X` sets - CRITICAL FOR CONSISTENCY):**
    * Apply to categorical columns of `X_train_imputed2` (or the scaled version if applicable, though scaling is usually for numerical only).
    * Initialize `OneHotEncoder` (`handle_unknown='ignore'`, `sparse_output=False`).
    * **Fit `OneHotEncoder` *ONLY* on categorical columns of `X_train_imputed2`.**
    * **Save as `one_hot_encoder.pkl`**.
    * Use it to **transform** categorical columns in `X_train_imputed2`, `X_val_imputed2`, and `X_test_imputed2`.
    * Combine all processed numerical (now scaled) and categorical (now encoded) features to form the final `X_train_final`, `X_val_final`, `X_test_final`. Ensure the date index is preserved.

9.  **Output Files:**
    * The script `feature.py` should save:
        * **Fitted Transformers:** `initial_numerical_imputer.pkl`, `initial_categorical_imputer.pkl`, `final_numerical_feature_imputer.pkl`, `final_categorical_feature_imputer.pkl`, `scaler.pkl`, `one_hot_encoder.pkl`.
        * **Processed Feature Sets (X):** `x_train.csv` (from `X_train_final`), `x_val.csv` (from `X_val_final`), `x_test.csv` (from `X_test_final`).
        * **Target Variables (y):** `y_train.csv`, `y_val.csv`, `y_test.csv` (corresponding to the rows remaining in X sets after NaN drops from signal generation).

**Important Considerations for Code Generation:**
* The script should use `pandas` for data manipulation and `sklearn` for transformations. Mention use of `TA-Lib` or `pandas_ta` if those are intended for technical indicators.
* All file operations use paths relative to the current directory.
* The script should be robust, print informative messages, and handle edge cases (e.g., no categorical columns, all data imputed in the initial step).

