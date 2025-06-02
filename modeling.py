def perform_modeling(
    modeling_script_filename,
    modeling_report_filename,
    x_train_path,
    y_train_path,
    x_val_path,
    y_val_path,
    x_test_path,
    y_test_path,
    model_output_path,
    scaler_path,
    preliminary_analysis_report_content,
    eda_result_summary,
    feature_engineering_report,
    date_index_name
):
    date_index_name_str = str(date_index_name) if date_index_name else 'None'

    modeling_prompt = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON MODELING INSTRUCTIONS**

**Your Mission:**
Your primary responsibility is to analyze provided data summaries and then construct a clear, concise, and actionable set of "Modeling Task Instructions." This set of instructions is your *sole output* and will be passed to an intelligent `pythonTool` to generate and execute Python code for a machine learning modeling pipeline.

**Inputs for Your Core Reasoning Process (Analyze these to decide on modeling strategy):**
1.  **`eda_report_content`**: "{eda_result_summary}"
2.  **`feature_engineering_report_content`**: "{feature_engineering_report}"
3.  **`preliminary_analysis_report_context`**: "{preliminary_analysis_report_content}"

**Your Decision-Making and Reasoning Process (Phases 1 & 2):**

**Phase 1: Analyze Report Data**
* Based on the content of `eda_report_content`, `feature_engineering_report_content`, and `preliminary_analysis_report_context`:
    * Outlier Assessment: Identify any features with significant outliers mentioned in `eda_report_content`. Decide on a specific handling strategy (e.g., "Apply winsorizing to 'column_x' with limits [0.01, 0.01]").
    * Skewness Assessment: Identify any highly skewed features mentioned in `eda_report_content`. Decide on a transformation strategy (e.g., "Apply np.log1p to 'column_y'").
    * Model Suitability: Note any hints from `feature_engineering_report_content` or overall context that suggest suitable model families.

**Phase 2: Formulate Modeling Strategy**
* A. Define Specific Preprocessing Steps: Consolidate your decisions from Phase 1 into a logical sequence of preprocessing actions for the `pythonTool`. These actions should include any necessary transformations, the use of a primary scaler (from the `scaler_path` that will be provided to the `pythonTool`), and any specific outlier treatments.
* B. Select Primary Model: Choose ONE primary regression model (e.g., RandomForestRegressor), justifying your choice based on the analyzed reports.
* C. Design Hyperparameter Tuning Plan: Specify the method (e.g., GridSearchCV), number of cross-validation folds (cv>=3), and the scoring metric (e.g., 'neg_root_mean_squared_error' or 'r2'). Crucially, define the hyperparameter grid for the chosen model as a Python dictionary string (e.g., "{{'n_estimators': [100, 200], 'max_depth': [10, 20]}}").

**Phase 3: Construct Your Output – The "Modeling Task Instructions" for `pythonTool`**
* Your final output MUST be a single string: the "Modeling Task Instructions". This string should be clearly formatted (e.g., using Markdown).
* The output string must begin with an overall **"Objective"**.
* Immediately following the "Objective", your output string must include a **"Configuration Details"** section. This section must clearly list all the following operational parameters with their respective actual values:
    * Script Filename Context: "{modeling_script_filename}"
    * Report Filename: "{modeling_report_filename}"
    * X_train Path: "{x_train_path}"
    * Y_train Path: "{y_train_path}"
    * X_val Path: "{x_val_path}"
    * Y_val Path: "{y_val_path}"
    * X_test Path: "{x_test_path}"
    * Y_test Path: "{y_test_path}"
    * Model Output Path: "{model_output_path}"
    * Scaler Path (pre-fitted): "{scaler_path}"
    * Date Index Name: "{date_index_name_str}"
* After the "Configuration Details" section, your output string must include a **"Tasks for pythonTool"** section. This should be a numbered list of sequential tasks.
* These tasks should clearly incorporate your reasoned modeling decisions from Phase 2 (preprocessing steps, chosen model, hyperparameter grid string, etc.) and instruct the `pythonTool` to refer to the values in the "Configuration Details" section for paths and filenames.

---
**GUIDANCE FOR THE FORMAT OF YOUR "MODELING TASK INSTRUCTIONS" OUTPUT STRING:**
---

`# Python Modeling Task Instructions`

`**Objective:** Build, train, evaluate, and save a regression model for stock price prediction.`

`**Configuration Details:**`
`    # [Prompt Creator: In YOUR OUTPUT string, list all configuration parameters here as specified in Phase 3 above. For example:]`
`    # * Script Filename Context: {actual value of modeling_script_filename}`
`    # * Report Filename: {actual value of modeling_report_filename}`
`    # * X_train Path: {actual value of x_train_path}`
`    # * ... and so on for all other configuration parameters ...`

`**Tasks for pythonTool:**`

`1.  **Load Data:**`
`    * Instruct the \`pythonTool\` to load datasets from the X and Y paths (as specified in the 'Configuration Details' section of your output) for train, validation, and test sets.`
`    * Instruct to use the 'Date Index Name' (from 'Configuration Details') as the index for all DataFrames, ensuring it's parsed as datetime.`
`    * Instruct to ensure the target variable \`y\` from each Y-path is prepared as a 1D array or pandas Series suitable for model training.`

`2.  **Preprocess Data:**`
`    * [Prompt Creator: Provide your sequence of specific, ordered preprocessing instructions decided in Phase 2A. These should be actionable steps for the \`pythonTool\`. Refer to specific column names if transformations or outlier treatments are column-specific. Instruct the \`pythonTool\` to use the 'Scaler Path' (from 'Configuration Details') for loading the pre-fitted scaler at the appropriate step in your sequence.]`
`    # Example instruction format: "1. Apply np.log1p transformation to the 'your_decided_column_for_log_transform' column in X_train, X_val, and X_test (if the column exists)." `
`    # Example instruction format: "2. Load the pre-fitted scaler from the 'Scaler Path'." `
`    # Example instruction format: "3. Apply the loaded scaler to the features of X_train, X_val, and X_test." `

`3.  **Train Model:**`
`    * Instruct the \`pythonTool\` to use the model: [Prompt Creator: Insert the name of the model you selected in Phase 2B, e.g., RandomForestRegressor].`
`    * Instruct to perform hyperparameter tuning using [Prompt Creator: Insert GridSearchCV or RandomizedSearchCV as decided in Phase 2C] with cv=[Prompt Creator: Insert CV fold number, e.g., 3] and scoring='[Prompt Creator: Insert scoring metric, e.g., neg_root_mean_squared_error]'.`
`    * Instruct that the hyperparameter grid to search is exactly: [Prompt Creator: Insert the Python dictionary string for the param_grid you formulated in Phase 2C, e.g., "{{'n_estimators': [100, 200], 'max_depth': [10, 20]}}"].`
`    * Instruct that the final model should be the best estimator found.`
`    * Instruct to log or print the best hyperparameters found.`

`4.  **Evaluate Model:**`
`    * Instruct the \`pythonTool\` to calculate and log/print MAE, MSE, RMSE, and R2 scores for the final model on the training, validation, and test sets.`

`5.  **Save Artifacts:**`
`    * Instruct the \`pythonTool\` to save the trained final model to the 'Model Output Path' (as specified in 'Configuration Details').`
`    * Instruct to generate a performance report as a text file, saving it to the 'Report Filename' (as specified in 'Configuration Details'). This report must contain:`
`        * Chosen model name (from your Phase 2B decision).`
`        * Best hyperparameters found.`
`        * Clearly tabulated MAE, MSE, RMSE, and R2 scores for training, validation, and test sets.`
`        * Placeholder sections for qualitative analysis: "Overfitting/Underfitting Analysis:", "Model Robustness:", and "Suggestions for Next Steps:".`

`End of Modeling Task Instructions.`
"""
    return modeling_prompt
