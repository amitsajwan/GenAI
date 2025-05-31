**Goal:**
Generate a Python script named `feature.py` to perform feature engineering on the provided cleaned datasets. The script should first create new domain-specific features (market signals), then handle missing values in all features, perform feature scaling, and encode categorical features consistently across train, validation, and test sets. It must also separate the target variable and save all resulting datasets and fitted transformers. Ensure the date index present in the input files is preserved in all output feature DataFrames.

**Input Files (located in the current directory):**
1.  `Cleaned_Train.csv` (training data)
2.  `Cleaned_Val.csv` (validation data)
3.  `Cleaned_Test.csv` (test data)
4.  `eda_result_summary.txt` (for insights, if helpful)
5.  `preliminary_analysis_report.txt` (for insights, if helpful)
    *(These input files are expected to contain columns like 'Open', 'High', 'Low', 'Close', 'Volume' or similar, in addition to any other existing features.)*

**Target Variable:**
The target variable to be separated is named 'Close'.

**Date Index:**
The input CSV files have a date index (assume it's named 'numeric_date_idx' or infer from data if no specific name given in reports). This index MUST be preserved in all output feature DataFrames (`x_train.csv`, `x_val.csv`, `x_test.csv`).

**Core Tasks for `feature.py` (in order):**

1.  **Load Data:**
    * Load `Cleaned_Train.csv`, `Cleaned_Val.csv`, and `Cleaned_Test.csv` into pandas DataFrames, ensuring the date index is correctly set.

2.  **Separate Target Variable:**
    * From each DataFrame (train, val, test), separate the target variable ('Close') into `y_train`, `y_val`, and `y_test`.
    * The remaining columns will form `X_train_raw`, `X_val_raw`, and `X_test_raw`.

3.  **Initial Missing Value Imputation (for existing raw features BEFORE creating new time-series features):**
    * This step is to ensure that calculations for technical indicators have complete data to work with from the original features.
    * **Identify Column Types:** Distinguish between numerical and categorical columns in `X_train_raw`.
    * **Numerical Imputation:**
        * Initialize and fit a `SimpleImputer` (e.g., strategy 'median' or 'ffill' for time series) *ONLY* on numerical columns of `X_train_raw`. Save as `initial_numerical_imputer.pkl`.
        * Transform numerical columns in `X_train_raw`, `X_val_raw`, and `X_test_raw`.
    * **Categorical Imputation:**
        * Initialize and fit a `SimpleImputer` (e.g., strategy 'most_frequent') *ONLY* on categorical columns of `X_train_raw`. Save as `initial_categorical_imputer.pkl`.
        * Transform categorical columns in `X_train_raw`, `X_val_raw`, and `X_test_raw`.
    * Let the results be `X_train_imputed1`, `X_val_imputed1`, `X_test_imputed1`.

4.  **Domain-Specific Feature Creation (Market Signals):**
    * Using the `X_train_imputed1`, `X_val_imputed1`, and `X_test_imputed1` DataFrames (which should contain 'Open', 'High', 'Low', 'Close', 'Volume' columns or similar from the input files):
    * Generate a set of common technical indicators and market signals. You can use libraries like `pandas` for rolling calculations, or if the execution environment supports it, `TA-Lib` or `pandas_ta` (if using these, include necessary imports). Consider features such as:
        * Moving Averages: e.g., SMA_7, SMA_21, EMA_14 for the 'Close' price (or another relevant price column).
        * Lag Features: e.g., 'Close_lag_1', 'Close_lag_3', 'Volume_lag_1'.
        * Rate of Change (ROC): e.g., for 'Close' price over 1 day or N days.
        * Relative Strength Index (RSI): e.g., for a 14-day period.
        * Moving Average Convergence Divergence (MACD): With common parameters.
        * Bollinger Bands: Upper, Middle (which is an SMA), and Lower bands.
        * Volatility: e.g., rolling standard deviation of 'Close' price returns over N days.
        * Other indicators: Stochastics, Average True Range (ATR), etc.
    * Choose appropriate window sizes for these indicators (e.g., 7, 14, 21, 30, 50 days) or use insights from the EDA reports if available to guide your choices.
    * Append these new features to `X_train_imputed1`, `X_val_imputed1`, and `X_test_imputed1`. Let the results be `X_train_with_signals`, `X_val_with_signals`, `X_test_with_signals`.
    * **Handle NaNs Introduced by New Features:** Creating lag features or rolling window features will introduce NaNs at the beginning of each series. **After generating all new features, drop any rows that now contain NaNs resulting from these calculations.** This should be done consistently across `X_train_with_signals` (and its corresponding `y_train`), `X_val_with_signals` (and `y_val`), and `X_test_with_signals` (and `y_test`) to ensure all data fed to the model is complete and aligned. Be careful to drop corresponding rows from `y_train`, `y_val`, `y_test` as well. After dropping, re-assign to `X_train`, `X_val`, `X_test` and their corresponding `y` variables.

5.  **Final Missing Value Imputation (CRITICAL FOR CONSISTENCY - for any remaining NaNs in *all* features):**
    * This step handles any NaNs that might still exist or were introduced by features not perfectly handled by the previous drop (e.g., from categorical features or if some indicators had internal NaNs not at the start).
    * **Identify Column Types:** Automatically distinguish between numerical and categorical columns in the current `X_train`.
    * **Numerical Imputation:**
        * Initialize a `SimpleImputer` (e.g., 'median').
        * **Fit this numerical imputer *ONLY* on the numerical columns of the current `X_train`.**
        * **Save as `final_numerical_imputer.pkl`**.
        * Use it to **transform** numerical columns in `X_train`, `X_val`, and `X_test`.
    * **Categorical Imputation:**
        * Initialize a `SimpleImputer` (e.g., 'most_frequent').
        * **Fit this categorical imputer *ONLY* on the categorical columns of the current `X_train`.**
        * **Save as `final_categorical_imputer.pkl`**.
        * Use it to **transform** categorical columns in `X_train`, `X_val`, and `X_test`.
    * Combine imputed columns back. Let the results be `X_train_imputed2`, `X_val_imputed2`, `X_test_imputed2`.

6.  **Feature Scaling (Numerical Features - CRITICAL FOR CONSISTENCY):**
    * Apply this step *after all imputation and new feature creation*.
    * Initialize a `StandardScaler`.
    * **Fit the `StandardScaler` *ONLY* on the numerical columns of `X_train_imputed2`.**
    * **Save as `scaler.pkl`**.
    * Use it to **transform** numerical columns in `X_train_imputed2`, `X_val_imputed2`, and `X_test_imputed2`.
    * Let the results be `X_train_scaled`, `X_val_scaled`, `X_test_scaled`.

7.  **Categorical Feature Encoding (CRITICAL FOR CONSISTENCY):**
    * Apply this step *after all imputation*.
    * Identify remaining categorical columns in `X_train_imputed2`.
    * Initialize a `OneHotEncoder` (set `handle_unknown='ignore'`, `sparse_output=False`).
    * **Fit it *ONLY* on the categorical columns of `X_train_imputed2`.**
    * **Save as `one_hot_encoder.pkl`**.
    * Use it to **transform** categorical columns in `X_train_imputed2`, `X_val_imputed2`, and `X_test_imputed2`.
    * Concatenate newly encoded features with the numerical (and scaled) features to form the final `X_train_final`, `X_val_final`, `X_test_final`. Ensure date index is preserved.

8.  **Output Files:**
    * The script `feature.py` should save:
        * **Fitted Transformers:** `initial_numerical_imputer.pkl`, `initial_categorical_imputer.pkl`, `final_numerical_imputer.pkl`, `final_categorical_imputer.pkl`, `scaler.pkl`, `one_hot_encoder.pkl`.
        * **Processed Feature Sets (X):** `x_train.csv` (from `X_train_final`), `x_val.csv` (from `X_val_final`), `x_test.csv` (from `X_test_final`).
        * **Target Variables (y):** `y_train.csv`, `y_val.csv`, `y_test.csv` (these are the versions after potential row drops in step 4).

**Important Considerations for Code Generation:**
* The script should use `pandas` for data manipulation and `sklearn` for transformations. If `TA-Lib` or `pandas_ta` are suggested for technical indicators, ensure the agent notes them as dependencies.
* All file operations use paths relative to the current directory.
* The script should be robust, print informative messages, and handle edge cases (e.g., no categorical columns, all data imputed in the initial step).

This revised prompt now explicitly guides GPT-4o to not only preprocess existing features but also to **create new, domain-relevant market signals**, and then handle all data consistently. The two-stage imputation (initial before signal generation, final after signal generation and NaN-row-dropping) offers more robustness.
