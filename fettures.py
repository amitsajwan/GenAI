# Inside your perform_feature_engineering function in Python:
# eda_report_content_actual_value = "..." # This comes from your previous steps
# preliminary_analysis_report_content_actual_value = "..." # This comes from your previous steps
# cleaned_train_path_actual_value = fe_script_args['cleaned_train_path'] # Example mapping
# ... and so on for all necessary arguments ...

fe_prompt_for_creator_llm = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON FEATURE ENGINEERING SCRIPT GENERATION**

**Your Mission:**
Your primary responsibility is to analyze the provided data summaries and configuration details. Based on this analysis, you will make strategic decisions for feature engineering. Your *sole output* will be a highly detailed and explicit "Execution Prompt" designed for a non-reasoning `pythonTool`. This `pythonTool` will use your "Execution Prompt" to generate a Python script that performs the feature engineering tasks you've outlined, preparing data for modeling.

**SECTION 1: CONTEXT AND DATA (Values embedded by the calling system)**

1.  **EDA Report Summary (`eda_report_content`):**
    ```
    {eda_report_content_actual_value}
    ```
2.  **Preliminary Analysis Report Context (`preliminary_analysis_report_content`):**
    ```
    {preliminary_analysis_report_content_actual_value}
    ```
3.  **Input Data Paths:**
    * Training Data (cleaned): `{cleaned_train_path_actual_value}`
    * Validation Data (cleaned): `{cleaned_val_path_actual_value}` (Note: Will be 'None' or empty if not applicable)
    * Test Data (cleaned): `{cleaned_test_path_actual_value}` (Note: Will be 'None' or empty if not applicable)
4.  **Output Data Paths (Targets for the `pythonTool` script):**
    * Feature-Engineered Training Data: `{fe_train_output_path_target_actual_value}`
    * Feature-Engineered Validation Data: `{fe_val_output_path_target_actual_value}`
    * Feature-Engineered Test Data: `{fe_test_output_path_target_actual_value}`
    * Fitted Feature Transformers (e.g., imputers, encoders, scalers): `{feature_transformer_output_path_target_actual_value}`
5.  **Key Column Names:**
    * Target Variable: `{target_variable_name_actual_value}`
    * Index Column: `{index_name_actual_value}`
    * OHLCV Columns (if applicable, comma-separated): `{ohlcv_columns_actual_value}`
    * Other Date Columns (if applicable, comma-separated): `{date_columns_list_actual_value}`
6.  **Target Script and Report Names (for the `pythonTool` script):**
    * Feature Engineering Script Filename: `{target_fe_script_filename_actual_value}`
    * Feature Engineering Report Filename: `{target_fe_report_filename_actual_value}`

**SECTION 2: YOUR STRATEGIC DECISION-MAKING FOR FEATURE ENGINEERING**
(Based on the inputs in Section 1, particularly `eda_report_content` and `preliminary_analysis_report_content`, determine the feature engineering strategy. Your decisions here will guide the structure and content of the "Execution Prompt" you generate for the `pythonTool`.)

1.  **Column Identification and Grouping:**
    * Based on EDA and provided column lists, what are the key numerical, categorical, date, and OHLCV columns to be processed?
2.  **Initial Imputation (Full DataFrame, if necessary):**
    * Is an initial imputation needed (e.g., for 'Close' price if it's used in early signal generation before X/y split)? What strategy (e.g., forward-fill, interpolation)?
3.  **Domain-Specific Feature/Signal Generation (e.g., Market Signals):**
    * What new features specific to the domain (e.g., lagged features, moving averages, volatility, financial ratios, interaction terms based on `preliminary_analysis_report_content` or EDA) should be created? List the types and key parameters (e.g., lag periods, window sizes).
4.  **Post-Signal NaN Handling:**
    * After creating new signals, how should introduced NaNs (typically at the start of series) be handled (e.g., drop rows, specific imputation)? Ensure chronological alignment.
5.  **Target Variable Separation:**
    * At what point should the target variable (`{target_variable_name_actual_value}`) be separated from the feature set? (Usually after all features that might use it, like 'Close' for creating returns, are generated).
6.  **Final Imputation (X features only):**
    * After target separation, what imputation strategy is needed for remaining NaNs in numerical X features (e.g., median, mean)?
    * What imputation strategy for categorical X features (e.g., mode, constant 'missing')?
7.  **Feature Scaling (Numerical X features):**
    * What scaling technique (e.g., StandardScaler, MinMaxScaler, RobustScaler) should be applied to numerical X features?
8.  **Categorical Encoding (Categorical X features):**
    * What encoding scheme (e.g., OneHotEncoder with `handle_unknown='ignore'`, TargetEncoder if appropriate) for categorical X features? Consider cardinality.
9.  **Feature Dropping (Optional):**
    * Are there any original or intermediate columns to be dropped after they've served their purpose?
10. **Order of Operations:**
    * Outline the logical sequence of these FE steps.

**SECTION 3: CONSTRUCT YOUR OUTPUT – THE "EXECUTION PROMPT" FOR `pythonTool`**
(Your *sole output* for this current task is a single string: the "Execution Prompt." This "Execution Prompt" must be a valid Python f-string (it must start with `f\"\"\"` and end with `\"\"\"`). It will instruct the `pythonTool` to generate a Python script that implements the feature engineering strategy you formulated in Section 2.)

**Key Requirements for the "Execution Prompt" you generate:**

* **Clarity and Detail:** It must provide unambiguous instructions for the `pythonTool` to generate a Python script.
* **Structure:** Guide the `pythonTool` to create a script with clear sections or functions for each major FE step (loading, imputation, signal generation, NaN handling, target separation, scaling, encoding, saving data, saving transformers, generating a report).
* **Fit/Transform Paradigm:** Emphasize that any fittable transformer (imputer, scaler, encoder) must be `fit` ONLY on the training data (`X_train`) and then used to `transform` the training, validation, and test sets.
* **Saving Artifacts:** Instruct the `pythonTool` to ensure its generated script saves:
    * The transformed DataFrames (train, val, test) to the paths specified in Section 1.
    * All *fitted* transformers (imputers, scalers, encoders) to a file or files related to `{feature_transformer_output_path_target_actual_value}` using `joblib`.
    * A simple text report to `{target_fe_report_filename_actual_value}` summarizing FE steps and output data shapes.
* **Variable Embedding:** The "Execution Prompt" you generate must correctly embed the *actual string values* for paths and filenames (e.g., "{cleaned_train_path_actual_value}", "{target_fe_script_filename_actual_value}") into its instructions for the `pythonTool`.
* **Pythonic Code:** Guide the `pythonTool` towards generating clean, readable Python.

**Example of how you (Prompt Creator LLM) might instruct the `pythonTool` for a specific step within the Execution Prompt you are generating:**
"...For the 'Market Signal Generation' part of the Execution Prompt, instruct the `pythonTool` as follows:
`# --- Market Signal Generation ---`
`# Based on the EDA and preliminary analysis, the following signals were identified as potentially useful:`
`# 1. Create 5-day and 20-day Simple Moving Averages (SMA) for the '{ohlcv_columns_actual_value}'.split(',')[3] column (assuming 'Close').`
`#    Example for 5-day SMA on 'Close': df['SMA_5_Close'] = df['Close'].rolling(window=5).mean()`
`# 2. Create a 1-day lag for the '{ohlcv_columns_actual_value}'.split(',')[3] column.`
`#    Example: df['Close_Lag1'] = df['Close'].shift(1)`
`# Apply these to df_train, df_val, and df_test consistently...`
This way, you provide strategic guidance on *what* signals, and the LLM details *how* to instruct the `pythonTool` to code it."
"""

    # In your Python code, after this fe_prompt_for_creator_llm is defined and populated:
    # execution_prompt_for_fe_tool = call_llm(fe_prompt_for_creator_llm)
    # ... then pass execution_prompt_for_fe_tool to your agent with pythonTool ...
    # return execution_prompt_for_fe_tool # Or directly use it
