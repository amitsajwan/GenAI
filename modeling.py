perform_modeling(modeling_script_filename, modeling_report_filename, x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, y_test_path, model_output_path, scaler_path, date_index_name, eda_result_summary_content, feature_engineering_report_content) # Assuming these are all passed
# generate the dynamic prompt for modeling
modeling_dynamic_prompt = f"""
**Goal:** Generate an executable Python script ('{modeling_script_filename}') to build, train, evaluate, and save a machine learning regression model for stock price prediction. The script should prioritize robustness and generalization.

**Overall Objective:** Build and train a machine learning model using '{{x_train_path}}', '{{y_train_path}}', '{{x_val_path}}', '{{y_val_path}}', '{{x_test_path}}', and '{{y_test_path}}'.

**Rules for Python code:**
* Program accepts no arguments; all inputs are static paths/values provided in this prompt.
* Do not suppress exceptions; let them propagate.
* Use Joblib for model save/load.
* All pkl / models should be read or created under a 'models' folder (implicitly handled by '{{model_output_path}}' and '{{scaler_path}}').

**Python Script Inputs (for '{modeling_script_filename}'):**
* 'x_train_path': String, path to '{x_train_path}'.
* 'y_train_path': String, path to '{y_train_path}'.
* 'x_val_path': String, path to '{x_val_path}'.
* 'y_val_path': String, path to '{y_val_path}'.
* 'x_test_path': String, path to '{x_test_path}'.
* 'y_test_path': String, path to '{y_test_path}'.
* 'model_output_path': String, path to save the trained model '{model_output_path}'.
* 'scaler_path': String, path to the pre-fitted scaler '{scaler_path}'.
* 'date_index_name': String, name of the date index column '{date_index_name}'.

**Data Insights (Derived from EDA & Feature Engineering - these should heavily influence the generated script):**
* EDA Result Summary (Content Embedded):
    {{eda_result_summary_content}}
* Feature Engineering Analysis Content (Embedded):
    {{feature_engineering_report_content}}
* **Key actionable insights from EDA to be incorporated into the Python script:**
    * **Outlier Impact:** Based on the EDA, note if significant outliers were detected that might disproportionately affect RMSE.
    * **Feature Distributions:** Note any highly skewed features that might benefit from transformation.
    * **Feature Correlations:** Note high multicollinearity if observed.

**Python Script Instructions (for '{modeling_script_filename}'):**

**1. Load Data:**
* Load '{{x_train_path}}', '{{y_train_path}}', '{{x_val_path}}', '{{y_val_path}}', '{{x_test_path}}', '{{y_test_path}}' into pandas DataFrames.
* Ensure '{{date_index_name}}' is set as the index for all loaded DataFrames if it's part of the CSVs.

**2. Preprocess Data:**
* Load the primary fitted scaler from '{{scaler_path}}' (e.g., StandardScaler, MinMaxScaler).
* Apply this primary scaler to X_train, X_val, and X_test.
* **Advanced Preprocessing based on Data Insights:**
    * **If '{{eda_result_summary_content}}' indicates significant, impactful outliers for specific features:**
        * The generated Python script should consider implementing an additional outlier handling step *after* initial scaling if the primary scaler is sensitive (e.g. StandardScaler), or *before* if using a scaler like MinMaxScaler. Options for the script:
            * Apply `RobustScaler` to the identified features if not the primary scaler.
            * Implement outlier capping/clipping (e.g., winsorizing at 1st and 99th percentiles) for features explicitly identified as problematic in EDA.
    * **If '{{eda_result_summary_content}}' indicates highly skewed distributions for important features:**
        * The generated Python script should consider applying transformations like log transform (`np.log1p`) or Box-Cox transform to these features (ensure to handle transformations consistently across train/val/test and potentially inverse transform predictions if target is transformed).

**3. Model Selection:**
* Based on insights from '{{eda_result_summary_content}}', '{{feature_engineering_report_content}}', and general best practices for stock data (which can be prone to overfitting), select an appropriate regression model.
* Candidate models: Linear Regression (as a baseline), Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor.
* **Justification for model choice should be included in the Model Evaluation Report.** Prioritize models that offer good control over complexity.

**4. Model Training & Hyperparameter Tuning:**
* Train the selected model using the preprocessed X_train and y_train.
* **Crucially, the generated Python script MUST include hyperparameter tuning for the selected model (unless it's a simple model like Linear Regression without tunable hyperparameters beyond fit_intercept). Use scikit-learn's `GridSearchCV` or `RandomizedSearchCV` with at least 3-fold cross-validation on the training data (preprocessed X_train, y_train).**
* The tuning process should aim to optimize for a robust metric like 'neg_root_mean_squared_error' or 'r2'.
* **If Random Forest Regressor is selected, the hyperparameter tuning grid in the generated script MUST include (but is not limited to):**
    * `n_estimators`: [50, 100, 200]
    * `max_depth`: [3, 5, 7, 10, 15]
    * `min_samples_split`: [2, 5, 10, 20]
    * `min_samples_leaf`: [1, 2, 4, 8, 12]
    * `max_features`: ['sqrt', 'log2', 0.7, 0.8]
* **If Gradient Boosting or XGBoost Regressor is selected, include a similar comprehensive tuning grid relevant to those models (e.g., learning_rate, n_estimators, max_depth, subsample, colsample_bytree).**
* The generated script must use the **best model (best_estimator_ attribute from GridSearchCV/RandomizedSearchCV)** found during hyperparameter tuning for subsequent validation and testing.
* Log the best hyperparameters found.

**5. Model Validation:**
* Validate the **tuned model** (best_estimator_) using the preprocessed X_val and y_val.
* Calculate and log/print the following metrics for the **validation set**:
    * Mean Absolute Error (MAE)
    * Mean Squared Error (MSE)
    * Root Mean Squared Error (RMSE)
    * R-squared (R²)
* Also, calculate and log/print the same metrics for the **training set** using the tuned model to compare against validation metrics and assess overfitting.

**6. Model Testing:**
* Test the **final tuned model** on the preprocessed X_test and y_test.
* Calculate and log/print the same performance metrics (MAE, MSE, RMSE, R²) for the **test set**.

**7. Save the Model:**
* Save the **final tuned model** (the best_estimator_ from hyperparameter tuning) to the specified '{{model_output_path}}' using `joblib.dump()`.

**8. Prepare Model Evaluation Report (content to be written to '{modeling_report_filename}'):**
* **Chosen Model and Justification:** Why was this model architecture selected based on data insights or problem type?
* **Hyperparameters:**
    * If tuned: List the best hyperparameters found and the range/values searched.
    * If not tuned: List default or chosen parameters.
* **Performance Metrics (clearly table or list them):**
    * Training Set: MAE, MSE, RMSE, R² (from the final tuned model)
    * Validation Set: MAE, MSE, RMSE, R² (from the final tuned model)
    * Test Set: MAE, MSE, RMSE, R² (from the final tuned model)
* **Observations and Insights:**
    * **Overfitting/Underfitting Analysis:** Explicitly compare training, validation, and test performance. Discuss the degree of overfitting or underfitting. A large gap between training and validation/test R² (e.g., train R² = 0.99, validation R² = 0.5) indicates overfitting.
    * **Model Robustness:** How well did the model generalize from validation to test?
    * **Comparison to Baseline:** How does the test R² compare to 0? (A negative R² means the model is worse than predicting the mean).
    * **Impact of Outlier Handling/Transformations (if applied):** Briefly comment if these steps were included and their perceived impact.
    * **Potential Reasons for Performance:** If performance is not satisfactory (e.g., low R², high RMSE), suggest potential reasons.
    * **Suggestions for Next Steps:** E.g., further feature engineering, trying different model architectures, collecting more data, more advanced outlier treatments.

**Python Script Outputs (Files to be created by '{modeling_script_filename}'):**
1.  '{model_output_path}': Trained machine learning model saved as a .pkl file.
2.  '{modeling_report_filename}': Text file containing the Model Evaluation Report as detailed above.

**Verification function (Logic to be included at the end of '{modeling_script_filename}' or as a separate check):**
* Verify that file '{modeling_report_filename}' is created.
* Verify that file '{model_output_path}' is created.
* Load the trained model from '{model_output_path}' and ensure it can make predictions on a small sample from X_test (e.g., first 5 rows) without error.
* Print a status message: "Model training, evaluation, and saving completed successfully. Report generated." or appropriate error messages.
"""

# The rest of your code to use this `modeling_dynamic_prompt`
# ...

# Example of how you might use the embedded content (ensure these variables are populated before creating the f-string)
# eda_result_summary_content = "Significant outliers found in 'Volume'. 'Close' price is right-skewed."
# feature_engineering_report_content = "Lagged features created. Rolling averages show predictive potential."

# perform_modeling(
#     modeling_script_filename="model_script.py",
#     modeling_report_filename="model_report.txt",
#     # ... other paths ...
#     eda_result_summary_content=eda_result_summary_content,
#     feature_engineering_report_content=feature_engineering_report_content
# )
