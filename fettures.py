# Inside your Python function perform_feature_engineering:
# The arguments like fe_script_filename, eda_result_summary, cleaned_train_path, etc.,
# will be directly used to populate the placeholders where their names match.
# For other critical info not in your function signature (like target_variable_name),
# you'll need to define them in your function and ensure they populate the
# {..._CONFIG} placeholders.

# Example:
# target_variable_name_value = "my_target_column" # Get this from config or define
# feature_transformer_output_path_value = "models/fe_transformers.pkl" # Define this path
# ohlcv_columns_value = "Open,High,Low,Close,Volume" # Or "" if not applicable
# other_date_columns_value = "EventDate" # Or "" if not applicable

fe_prompt_for_creator_llm = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON FEATURE ENGINEERING SCRIPT GENERATION**

**Your Mission:**
Your primary responsibility is to analyze the provided data summaries and configuration details. Based on this analysis, you will make strategic decisions for feature engineering. Your *sole output* will be a highly detailed and explicit "Execution Prompt" designed for a non-reasoning `pythonTool`. This `pythonTool` will use your "Execution Prompt" to generate a Python script that performs the feature engineering tasks you've outlined, preparing data for modeling.

**SECTION 1: CONTEXT AND DATA (Values embedded from your Python function's arguments and configurations)**

1.  **EDA Report Summary:**
    ```
    {eda_result_summary}
    ```
2.  **Preliminary Analysis Report Context:**
    ```
    {preliminary_analysis_report_content}
    ```
3.  **Input Data Paths (Cleaned X Features - FE process will also need corresponding y input):**
    * Training Features (X_train_cleaned): `{cleaned_train_path}`
    * Validation Features (X_val_cleaned): `{cleaned_val_path if cleaned_val_path else 'None'}`
    * Test Features (X_test_cleaned): `{cleaned_test_path if cleaned_test_path else 'None'}`
    *(LLM Note: Assume corresponding y data for these X inputs will be loaded by the FE script, e.g., by convention or if y paths are implicitly handled alongside these X paths in the source data structure. The `TARGET_VARIABLE_NAME_CONFIG` below is crucial for identifying and separating `y` if it's part of these input files.)*

4.  **Output Data Paths & Artifacts (Targets for the `pythonTool` script):**
    * Feature-Engineered Training Features (X_train_fe): `{x_train_path}`
    * Aligned Training Target (y_train_fe): `{y_train_path}`
    * Feature-Engineered Validation Features (X_val_fe): `{x_val_path}`
    * Aligned Validation Target (y_val_fe): `{y_val_path}`
    * Feature-Engineered Test Features (X_test_fe): `{x_test_path}`
    * Aligned Test Target (y_test_fe): `{y_test_path}`
    * Fitted Feature Transformers: `{{{feature_transformer_output_path_value}}}` *(This placeholder needs a value from your Python code, e.g., a predefined path)*

5.  **Key Column Names & Configuration:**
    * Target Variable Name: `{{{target_variable_name_value}}}` *(This placeholder needs a value from your Python code)*
    * Index Column Name: `{index_name}`
    * OHLCV Columns (comma-separated, or 'None'): `{{{ohlcv_columns_value}}}` *(This placeholder needs a value from your Python code)*
    * Other Date Columns (comma-separated, or 'None'): `{{{other_date_columns_value}}}` *(This placeholder needs a value from your Python code)*

6.  **Target Script and Report Names (for the `pythonTool` script):**
    * Feature Engineering Script Filename: `{fe_script_filename}`
    * Feature Engineering Report Filename: `{fe_report_filename}`

**SECTION 2: YOUR STRATEGIC DECISION-MAKING FOR FEATURE ENGINEERING**
(Based on the inputs in Section 1, particularly `eda_result_summary` and `preliminary_analysis_report_content`, determine the feature engineering strategy. Your decisions here will guide the structure and content of the "Execution Prompt" you generate for the `pythonTool`.)

1.  **Column Identification and Grouping:**
    * Based on EDA and provided column lists (OHLCV, Other Dates), identify numerical, categorical, date columns present in the input data (e.g., from `{cleaned_train_path}`). Note the `{{{target_variable_name_value}}}` and `{index_name}`.
2.  **Initial Imputation (Full DataFrame, if necessary, especially if target is used for FE):**
    * Is an initial imputation needed on data loaded from `{cleaned_train_path}` (and corresponding y) before further processing or X/y split (if target is in X file)? What strategy?
3.  **Target Variable Handling:**
    * If the target `{{{target_variable_name_value}}}` is present in the DataFrames loaded from `cleaned_train_path` (etc.), decide when it should be separated into `y_train`, `y_val`, `y_test`. This is usually after all features that might depend on it (like 'Close' for returns) are created, but before applying transformations only to X.
4.  **Domain-Specific Feature/Signal Generation (e.g., Market Signals):**
    * Based on `preliminary_analysis_report_content`, EDA, and `{{{ohlcv_columns_value}}}`, what new features or signals should be created? List types and key parameters (lags, windows for MAs, volatility, returns, etc.).
5.  **Date/Time Feature Creation:**
    * For columns listed in `{{{other_date_columns_value}}}` (and potentially from `{{{ohlcv_columns_value}}}` if they are date-like or `index_name` if it's a date index), what date/time components should be extracted (Year, Month, DayOfWeek, IsWeekend, etc.)?
6.  **Interaction/Polynomial Feature Creation (Optional):**
    * Does EDA suggest any useful interaction terms (numerical*numerical, numerical*categorical_encoded) or polynomial features?
7.  **Post-Feature Creation NaN Handling (Mainly for X, but align y):**
    * After creating new features (signals, lags, rolling means), how should introduced NaNs be handled (e.g., drop rows from X and correspondingly from y; or specific imputation for X)? Prioritize chronological alignment.
8.  **Final Imputation (X features only, after target separation):**
    * What imputation strategy for remaining NaNs in numerical X features (e.g., median, mean)?
    * What imputation strategy for categorical X features (e.g., mode, constant 'missing')?
9.  **Feature Scaling (Numerical X features):**
    * What scaling technique (e.g., StandardScaler, MinMaxScaler, RobustScaler) for numerical X features?
10. **Categorical Encoding (Categorical X features):**
    * What encoding scheme (e.g., OneHotEncoder with `handle_unknown='ignore'`, TargetEncoder) for categorical X features?
11. **Feature Dropping (Optional):**
    * Any columns to be dropped from X after they've served their purpose?
12. **Order of Operations:**
    * Outline the logical sequence of these FE steps.

**SECTION 3: CONSTRUCT YOUR OUTPUT – THE "EXECUTION PROMPT" FOR `pythonTool`**
(Your *sole output* for this current task is a single string: the "Execution Prompt." This "Execution Prompt" must be a valid Python f-string (it must start with `f\"\"\"` and end with `\"\"\"`). It will instruct the `pythonTool` to generate a Python script that implements the feature engineering strategy you formulated in Section 2.)

**Key requirements for the "Execution Prompt" you generate:**

* **Clarity and Detail:** Provide unambiguous instructions for the `pythonTool`.
* **Structure:** Guide `pythonTool` to create a script with functions for major FE steps.
* **Fit/Transform Paradigm:** Transformers (`fit` ONLY on training data, then `transform` train, val, test).
* **Saving Artifacts:** Instruct `pythonTool`'s script to save:
    * Transformed X DataFrames to paths like `{x_train_path}` (from Section 1, now an output path).
    * Aligned y DataFrames to paths like `{y_train_path}` (from Section 1, now an output path).
    * All *fitted* transformers to `{{{feature_transformer_output_path_value}}}`.
    * A report to `{fe_report_filename}`.
* **Variable Usage:** The "Execution Prompt" you generate must correctly embed the *actual string values* for all paths and filenames (e.g., "{cleaned_train_path}", "{x_train_path}", "{fe_script_filename}", `{{{target_variable_name_value}}}`) into its instructions for the `pythonTool`.
* **Pythonic Code:** Guide towards clean Python.

**Example of how you (Prompt Creator LLM) might instruct the `pythonTool` for target separation within the Execution Prompt:**
"...The Execution Prompt should instruct `pythonTool` for 'Target Variable Separation':
`# --- Target Variable Separation ---`
`# Assuming '{cleaned_train_path}' (and val/test) initially contains features and the target '{target_variable_name_value}'.`
`# Separate y_train = df_train_processed.pop('{target_variable_name_value}')`
`# X_train = df_train_processed`
`# Similarly for validation and test sets if they also contain the target column.`
`# If y data was loaded separately, ensure it's aligned (especially if rows were dropped from X).`
"
"""

    # This `fe_prompt_for_creator_llm` is now ready to be sent to your "Prompt Creator" LLM
    # after your `perform_feature_engineering` function has defined:
    # - target_variable_name_value
    # - feature_transformer_output_path_value
    # - ohlcv_columns_value
    # - other_date_columns_value
    # and passed all its arguments to fill the f-string.

    return fe_prompt_for_creator_llm
