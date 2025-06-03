# Inside your Python function perform_feature_engineering:
# The arguments like fe_script_filename, eda_result_summary, cleaned_train_path, etc.,
# will be directly used to populate the placeholders below.

fe_prompt_for_creator_llm = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON FEATURE ENGINEERING SCRIPT GENERATION**

**Your Mission:**
Your primary responsibility is to analyze the provided data summaries and configuration details. Based on this analysis, you will make strategic decisions for feature engineering. Your *sole output* will be a highly detailed and explicit "Execution Prompt" designed for a non-reasoning `pythonTool`. This `pythonTool` will use your "Execution Prompt" to generate a Python script that performs the feature engineering tasks you've outlined, preparing data for modeling.

**SECTION 1: CONTEXT AND DATA (Values embedded from your Python function's arguments)**

1.  **EDA Report Summary:**
    ```
    {eda_result_summary} 
    ```
    *(This will be filled by the `eda_result_summary` argument of your `perform_feature_engineering` function)*

2.  **Preliminary Analysis Report Context:**
    ```
    {preliminary_analysis_report_content}
    ```
    *(This will be filled by the `preliminary_analysis_report_content` argument)*

3.  **Input Data Paths (Cleaned Data):**
    * Training Features: `{cleaned_train_path}`
    * Training Target: `{y_train_path}` *(Assuming this path is for the target corresponding to `cleaned_train_path`)*
    * Validation Features: `{cleaned_val_path}` *(Note: Your signature also had `x_val_path`, ensure this maps correctly. Assuming `cleaned_val_path` refers to validation features. Handle if 'None' or empty)*
    * Validation Target: `{y_val_path}` *(Handle if 'None' or empty)*
    * Test Features: `{cleaned_test_path}` *(Note: Your signature also had `x_test_path`, ensure this maps correctly. Assuming `cleaned_test_path` refers to test features. Handle if 'None' or empty)*
    * Test Target: `{y_test_path}` *(Handle if 'None' or empty)*

4.  **Output Data Paths & Artifacts (Targets for the `pythonTool` script - these would need to be passed to or defined within `perform_feature_engineering`):**
    * Feature-Engineered Training Data: `{{FE_TRAIN_OUTPUT_PATH_PLACEHOLDER}}` 
    * Feature-Engineered Validation Data: `{{FE_VAL_OUTPUT_PATH_PLACEHOLDER}}`
    * Feature-Engineered Test Data: `{{FE_TEST_OUTPUT_PATH_PLACEHOLDER}}`
    * Fitted Feature Transformers (e.g., imputers, encoders, scalers): `{{FEATURE_TRANSFORMER_OUTPUT_PATH_PLACEHOLDER}}`
    *(Note: The placeholders above with `{{...}}` indicate that your `perform_feature_engineering` function needs to supply these target paths for the LLM to use when it constructs the prompt for the `pythonTool`. They are not directly in your visible function signature from the image, so you'd add them as arguments or define them.)*

5.  **Key Column Names & Configuration (These also need to be available to `perform_feature_engineering`):**
    * Target Variable Name: `{{TARGET_VARIABLE_NAME_PLACEHOLDER}}`
    * Index Column Name (from your function): `{index_name}`
    * OHLCV Columns (if applicable, comma-separated): `{{OHLCV_COLUMNS_PLACEHOLDER}}`
    * Other Date Columns (if applicable, comma-separated): `{{DATE_COLUMNS_LIST_PLACEHOLDER}}`
    *(Note: For `TARGET_VARIABLE_NAME_PLACEHOLDER`, `OHLCV_COLUMNS_PLACEHOLDER`, `DATE_COLUMNS_LIST_PLACEHOLDER`, your `perform_feature_engineering` function would need to receive these as arguments or from a config file to pass them here.)*

6.  **Target Script and Report Names (for the `pythonTool` script):**
    * Feature Engineering Script Filename: `{fe_script_filename}`
    * Feature Engineering Report Filename: `{fe_report_filename}`

**(Rest of the prompt: SECTION 2 and SECTION 3 remain as previously designed, guiding the LLM's reasoning and its output construction based on these now correctly mapped inputs.)**
"""

# Example of how you'd populate some of the currently missing placeholders within your function:
# (These would ideally be arguments to perform_feature_engineering or from a config)
# fe_train_output_path_target_actual_value = "data/feature_engineered/fe_train.csv"
# feature_transformer_output_path_target_actual_value = "models/fe_transformers.pkl"
# target_variable_name_actual_value = "target_close_price"
# ohlcv_columns_actual_value = "Open,High,Low,Close,Volume"
# date_columns_list_actual_value = "another_date_column"

# Then, when you create the final prompt string inside perform_feature_engineering,
# you would replace these {{..._PLACEHOLDER}} style placeholders as well.
# For instance, you might build the section for output paths like this:
# output_paths_str_for_prompt = f"""
# 4.  Output Data Paths (Targets for the `pythonTool` script):
#     * Feature-Engineered Training Data: "{fe_train_output_path_target_actual_value}" 
#     * ... (and so on)
# """
# And then embed output_paths_str_for_prompt into the main fe_prompt_for_creator_llm.
# Or, more simply, add all these as arguments to your perform_feature_engineering function.
