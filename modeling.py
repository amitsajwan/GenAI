# Inside your perform_modeling Python function:
# Arguments like modeling_script_filename, x_train_path, feature_engineering_report, etc.,
# will be directly used to populate the placeholders below.

modeling_prompt_for_creator_llm = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON MODELING SCRIPT GENERATION (PIPELINE FOCUS)**

**Your Mission:**
Your primary responsibility is to analyze provided data summaries (especially the Feature Engineering report) and configuration details. Based on this analysis, you will make strategic decisions for constructing a modeling pipeline. Your *sole output* will be a highly detailed and explicit "Execution Prompt" designed for a non-reasoning `pythonTool`. This `pythonTool` will use your "Execution Prompt" to generate a Python script that defines, trains (with hyperparameter tuning), evaluates, and saves a scikit-learn `Pipeline` as a **single, deployable .pkl file**.

**SECTION 1: CONTEXT AND DATA (Values embedded from your Python function's arguments)**

1.  **Feature Engineering Report Summary (`feature_engineering_report`):** (CRUCIAL INPUT)
    ```
    {feature_engineering_report}
    ```
    *(This report must detail what transformations, including imputation, scaling, and encoding, were performed in the FE stage and describe the state of the data being passed to modeling via `{x_train_path}`. This determines what preprocessing, if any, is needed in the modeling pipeline.)*

2.  **EDA Report Summary (for general context, `eda_result_summary`):**
    ```
    {eda_result_summary}
    ```
3.  **Preliminary Analysis Report Context (`preliminary_analysis_report_content`):**
    ```
    {preliminary_analysis_report_content}
    ```
4.  **Input Data Paths (Feature-Engineered Data):**
    * Training Features: `{x_train_path}`
    * Training Target: `{y_train_path}`
    * Validation Features: `{x_val_path if x_val_path else 'None'}`
    * Validation Target: `{y_val_path if y_val_path else 'None'}`
    * Test Features: `{x_test_path if x_test_path else 'None'}`
    * Test Target: `{y_test_path if y_test_path else 'None'}`

5.  **Output Path for Modeling Pipeline (Single .pkl file):**
    * Fitted Pipeline Output Path: `{model_output_path}`

6.  **Target Script and Report Names (for the `pythonTool` script):**
    * Modeling Script Filename: `{modeling_script_filename}`
    * Modeling Report Filename: `{modeling_report_filename}`

7.  **Configuration:**
    * Index Column Name: `{date_index_name if date_index_name else 'None'}`
    * Original Scaler Path (for context, if mentioned in FE report as a specific type used): `{scaler_path}`
        *(Note to LLM: The primary goal is a new pipeline. If the FE report says data from `{x_train_path}` is already scaled using a method compatible with this scaler, then this modeling pipeline might not need its own scaler step. If FE did not scale, or if a final scaling step is desired here, you will define a new scaler instance within the pipeline.)*

**SECTION 2: YOUR STRATEGIC DECISION-MAKING FOR THE MODELING PIPELINE**
(Based on the inputs in Section 1, particularly `feature_engineering_report`, formulate the modeling pipeline strategy.)

**Phase 1: Analyze `feature_engineering_report` and Define Modeling Pipeline Scope**
* **Assess Incoming Data State:** Based *critically* on the `feature_engineering_report`:
    * Is the data from `{x_train_path}` already fully imputed?
    * Is it already scaled? If yes, what was the method? Is it sufficient, or should this modeling pipeline apply its own final scaling?
    * Are categorical features (if any remain) already numerically encoded?
* **Determine Necessary Preprocessing Steps for *this* Modeling Pipeline:**
    * If the `feature_engineering_report` confirms data is fully preprocessed (scaled, encoded, imputed by FE), the modeling pipeline might only need the regressor step.
    * If scaling is *not* done by FE or if a final, consistent scaling step is preferred here: Decide on a scaler to instantiate in the pipeline (e.g., `StandardScaler`, `RobustScaler` based on EDA/FE insights about outliers or distributions of *engineered features*).
    * If final imputation is needed for any columns (e.g., a safeguard `SimpleImputer`): Decide on strategy.
    * If there are any remaining categorical features after FE that need encoding (less common if FE is comprehensive): Decide on an encoder.
    * The goal is a pipeline that robustly handles the data coming from your FE stage.

**Phase 2: Formulate Modeling Pipeline Strategy for `pythonTool`**
* **A. Define Pipeline Steps:**
    * Based on your Phase 1 determination, list the steps for the `sklearn.pipeline.Pipeline`. This could be very simple (e.g., just the regressor) or include a scaler/imputer if deemed necessary.
    * Example if FE output data is NOT yet scaled: `pipeline_steps = [('scaler', StandardScaler()), ('regressor', RandomForestRegressor(random_state=42))]`
    * Example if FE output data IS already scaled and encoded: `pipeline_steps = [('regressor', RandomForestRegressor(random_state=42))]`
* **B. Select Predictive Model:** Choose ONE primary regression model class (e.g., `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBoostRegressor`).
* **C. Design Hyperparameter Tuning Plan:**
    * Mandate `GridSearchCV` with `cv>=3` (e.g., `cv=5`).
    * Specify the scoring metric (e.g., 'neg_root_mean_squared_error', 'r2').
    * Instruct to define a relevant hyperparameter grid for the chosen model (e.g., `'regressor__n_estimators'`, `'regressor__max_depth'`). If any preprocessing steps (like a scaler) are included in *this* modeling pipeline and have tunable parameters, they can also be included in the grid (e.g., `'scaler__with_mean'`).

**SECTION 3: CONSTRUCT YOUR OUTPUT – THE "EXECUTION PROMPT" FOR `pythonTool`**
(Your *sole output* for this current task is a single string: the "Execution Prompt." This "Execution Prompt" must be a valid Python f-string (it must start with `f\"\"\"` and end with `\"\"\"`). It will instruct the `pythonTool` to generate a Python script that implements the pipeline strategy you formulated in Section 2.)

**Key requirements for the "Execution Prompt" you generate:**

1.  **Clarity and Detail:** Provide unambiguous instructions for the `pythonTool`.
2.  **Structure:** Guide the `pythonTool` to create a script with clear functions for:
    * Loading data (from `{x_train_path}`, etc. – this is the output of your FE stage).
    * Defining the `Pipeline` object with the steps you decided on in Section 2, Phase 2A.
    * Defining the hyperparameter grid for `GridSearchCV` as per Section 2, Phase 2C.
    * Training the pipeline using `GridSearchCV` on the input data. (The pipeline itself handles any internal scaling/transformations it's configured with).
    * Evaluating the best fitted pipeline on train, validation, and test sets.
    * Saving the single best fitted pipeline to `{model_output_path}`.
    * Generating a model evaluation report to `{modeling_report_filename}`.
3.  **Variable Usage:** The "Execution Prompt" must correctly embed the *actual string values* for paths and filenames (from Section 1) into its instructions for the `pythonTool`.
4.  **Data Handling:** Emphasize that the data loaded (output of FE) is fed directly to `GridSearchCV.fit()`.
5.  **Pythonic Code:** Guide towards clean, readable Python.

**Example of how you (Prompt Creator LLM) might instruct the `pythonTool` for defining the pipeline within the Execution Prompt you are generating:**
"...The Execution Prompt should then instruct the `pythonTool` to generate code for 'Define Pipeline and Hyperparameter Grid Function' like this:
`# --- Define Pipeline and Hyperparameter Grid Function ---`
`# def define_pipeline_and_grid():`
`#   pipeline_steps = []`
`#   # Based on your analysis of the FE report (Section 2, Phase 1 above):`
`#   # IF you determined scaling is still needed for data from {x_train_path}:`
`#   #   pipeline_steps.append(('scaler', StandardScaler())) # Or the scaler type you decided`
`#   # ELSE (if FE report confirms data is already scaled):`
`#   #   print("Data is assumed to be already scaled by the Feature Engineering stage. No scaler added to modeling pipeline.")`
`#   pipeline_steps.append(('[your_chosen_model_step_name_e.g._regressor]', [Your_chosen_model_class_instance_e.g.,_RandomForestRegressor(random_state=42)]))`
`#   pipeline = Pipeline(pipeline_steps)`
`#   param_grid = {{`
`#     # Add relevant hyperparameters for the steps included in pipeline_steps, e.g.:`
`#     # '[model_step_name]__n_estimators': [100, 200], '[model_step_name]__max_depth': [10, 20, None]`
`#     # If a scaler was added: `'scaler__with_mean': [True, False]`
`#   }}`
`#   return pipeline, param_grid`
This provides conditional logic for the `pythonTool` based on your reasoned decision."
"""

    # In your Python code, after this modeling_prompt_for_creator_llm is defined and populated:
    # execution_prompt_for_modeling_tool = call_llm(modeling_prompt_for_creator_llm)
    # ... then pass execution_prompt_for_modeling_tool to your agent with pythonTool ...
    # return execution_prompt_for_modeling_tool # Or directly use it
