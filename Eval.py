import os
import re
import json 
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END

# --- 1. Define the State for the Pipeline ---
class MultiAgentPipelineState(TypedDict):
    # Input
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 
    # Optional: hint about the problem type for evaluation metrics
    problem_type: Optional[Literal["classification", "regression"]] 

    # Output from EdaAgentNode
    eda_comprehensive_summary: Optional[str]
    eda_identified_issues: Optional[List[str]]
    eda_fe_suggestions: Optional[List[str]] 
    eda_processed_train_ref: Optional[str] 
    eda_processed_val_ref: Optional[str]   
    eda_processed_test_ref: Optional[str]  
    eda_plot_references: Optional[Dict[str, str]] 

    # Output from FeatureEngineeringAgentNode
    fe_applied_steps_summary: Optional[str]
    fe_final_feature_list: Optional[List[str]] 
    fe_X_train_transformed_ref: Optional[str] 
    fe_y_train_ref: Optional[str]
    fe_X_val_transformed_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_transformed_ref: Optional[str]  
    fe_transformer_references: Optional[Dict[str, str]] 
    fe_untrained_full_pipeline_ref: Optional[str] 

    # Output from ModelingNode
    model_trained_pipeline_ref: Optional[str] 
    model_training_summary: Optional[str]

    # Output from Evaluation Node
    evaluation_summary: Optional[str] # From evaluation agent
    evaluation_metrics: Optional[Dict[str, float]]
    test_set_prediction_status: Optional[str]


    # Control and tracking
    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    """
    This function is the integration point for your actual agno_python_tool.
    """
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction:\n    '{instruction}'")
    if agent_context_hint:
        print(f"    Agent Context Hint: {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    # EDA Related
    if "load the dataset from 'dummy_pipeline_data/train_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/train_data.csv' loaded. Tool refers to it as 'initial_train_df_ref'. Shape (3, 7)."
    elif "load the dataset from 'dummy_pipeline_data/val_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/val_data.csv' loaded. Tool refers to it as 'initial_val_df_ref'. Shape (3, 7)."
    elif "load the dataset from 'dummy_pipeline_data/test_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/test_data.csv' loaded. Tool refers to it as 'initial_test_df_ref'. Shape (3, 7)."
    elif "clean data referenced by 'initial_train_df_ref'" in instruction_lower:
        sim_observation += "Data 'initial_train_df_ref' cleaned. New reference is 'cleaned_train_df_eda.pkl'."
    elif "generate plot for 'price' distribution from 'cleaned_train_df_eda.pkl'" in instruction_lower:
        sim_observation += "Plot 'price_dist_plot_eda.png' generated for 'cleaned_train_df_eda.pkl'."
    
    # Feature Engineering Related
    elif "fit a standardscaler on data from 'cleaned_train_df_eda.pkl' for column 'price', save it, and report its reference" in instruction_lower:
        sim_observation += "StandardScaler fitted on 'Price' column of 'cleaned_train_df_eda.pkl'. Saved as 'fitted_price_scaler.pkl'. This is its reference."
    elif "create a scikit-learn pipeline with steps: scaler (ref: 'fitted_price_scaler.pkl'), and an untrained randomforestclassifier. save this untrained pipeline and report its reference." in instruction_lower:
        sim_observation += "Untrained Scikit-learn pipeline created with specified scaler and RandomForestClassifier. Saved as 'untrained_full_pipeline.pkl'. This is its reference."
    elif "separate target 'target' from features in 'cleaned_train_df_eda.pkl' and 'cleaned_val_df_eda.pkl'. report new references for x_train, y_train, x_val, y_val, and the processed x_test from 'cleaned_test_df_eda.pkl'." in instruction_lower:
        sim_observation += ("Target 'Target' separated. "
                           "Output data references: "
                           "X_train_ref: 'X_train_final_fe.pkl', y_train_ref: 'y_train_final_fe.pkl', "
                           "X_val_ref: 'X_val_final_fe.pkl', y_val_ref: 'y_val_final_fe.pkl', "
                           "X_test_ref: 'X_test_final_fe.pkl'. "
                           "Final feature list for X datasets: ['Price_scaled', 'Category_encoded_A', ...].")

    # Modeling Related
    elif "load the untrained pipeline from 'untrained_full_pipeline.pkl' and train it using x_train from 'x_train_final_fe.pkl' and y_train from 'y_train_final_fe.pkl'. save the trained pipeline and report its reference." in instruction_lower:
        sim_observation += "Untrained pipeline 'untrained_full_pipeline.pkl' loaded. Trained using 'X_train_final_fe.pkl' and 'y_train_final_fe.pkl'. Trained pipeline saved as 'trained_model_pipeline.pkl'. This is its reference."
    
    # Evaluation Related
    elif "load the trained pipeline from 'trained_model_pipeline.pkl'" in instruction_lower and "load x_val from 'x_val_final_fe.pkl' and y_val from 'y_val_final_fe.pkl'" in instruction_lower:
        sim_observation += "Trained pipeline 'trained_model_pipeline.pkl' loaded. Validation data X_val ('X_val_final_fe.pkl') and y_val ('y_val_final_fe.pkl') loaded."
    elif "make predictions on x_val using the loaded pipeline" in instruction_lower:
        sim_observation += "Predictions made on X_val. Predictions stored internally by tool as 'val_predictions'."
    elif "calculate classification metrics (accuracy, f1-score, roc auc) comparing 'val_predictions' against y_val" in instruction_lower:
        sim_observation += "Metrics calculated: Accuracy: 0.91, F1-Score: 0.89, ROC AUC: 0.95."
    elif "calculate regression metrics (mse, r-squared) comparing 'val_predictions' against y_val" in instruction_lower:
        sim_observation += "Metrics calculated: MSE: 15.2, R-squared: 0.78."
    elif "make predictions on x_test from 'x_test_final_fe.pkl' using the loaded pipeline" in instruction_lower:
        sim_observation += "Predictions made on X_test ('X_test_final_fe.pkl'). Predictions saved by tool to 'final_test_predictions.csv'."
    else:
        sim_observation += "Task completed. If new data artifacts or sklearn objects were created/saved, their references are included here or can be requested."
    
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1) 

def run_generic_react_loop(
    initial_prompt_content: str,
    max_iterations: int,
    agent_context_hint_for_tool: Optional[str] = None 
) -> str: 
    react_messages: List[BaseMessage] = [SystemMessage(content=initial_prompt_content)]
    final_answer_text = "Agent did not produce a Final Answer within iteration limit."
    
    for i in range(max_iterations):
        print(f"  [GenericReActLoop] Iteration {i+1}/{max_iterations}")
        ai_response = llm.invoke(react_messages)
        ai_content = ai_response.content.strip()
        react_messages.append(ai_response) 
        print(f"    LLM: {ai_content[:400]}...")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*Python\s*Action Input:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE) 

        if final_answer_match:
            final_answer_text = final_answer_match.group(1).strip()
            print(f"    Loop Concluded. Final Answer obtained.")
            break 
        elif action_match:
            nl_instruction_for_tool = action_match.group(1).strip()
            tool_observation = agno_python_tool_interface(nl_instruction_for_tool, agent_context_hint_for_tool)
            react_messages.append(HumanMessage(content=f"Observation: {tool_observation}"))
        else:
            react_messages.append(HumanMessage(content="System hint: Your response was not in the expected format. Please use 'Action: Python\\nAction Input: <NL_instruction>' or 'Final Answer: <summary>'."))
            if i > 1: 
                final_answer_text = "Agent failed to follow output format consistently."
                print(f"    {final_answer_text}")
                break 
        if i == max_iterations - 1: 
            print(f"    Max ReAct iterations reached.")
            if ai_content and not final_answer_match: 
                final_answer_text = f"Max iterations reached. Last AI thought: {ai_content}"
    return final_answer_text

# --- 4. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- EDA Agent Node Running ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'."

    prompt_content = f"""You are an Expert EDA Data Scientist. PythonTool takes NL instructions and reports data/plot references.
    Initial context for PythonTool: {eda_tool_context_hint}
    Instruct PythonTool to:
    1. Load train, val, test datasets using paths from context. Ask for their references.
    2. Using these references, check structure, quality (missing values, outliers - ask for plot refs), distributions (especially '{target_col}' - ask for plot ref), correlations (ask for plot ref).
    3. Perform initial cleaning if needed, ask for NEW references for cleaned datasets.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST include:
    1. EDA Comprehensive Summary: <details>
    2. Data Quality Report: (list issues like - Missing Values: <details>)
    3. Key Insights: (list insights like - Insight: <details>)
    4. Feature Engineering Suggestions: (list suggestions like - FE Suggestion: <details>)
    5. Data References (as reported by PythonTool):
       - Processed Train Data: <tool_reported_train_ref_after_eda_cleaning>
       - Processed Val Data: <tool_reported_val_ref_after_eda_cleaning>
       - Processed Test Data: <tool_reported_test_ref_after_eda_cleaning>
       - Plot - Target Distribution: <tool_reported_plot_ref> 
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 8), eda_tool_context_hint)
    
    parsed_output = {"current_stage_completed": "EDA"}
    summary_match = re.search(r"EDA Comprehensive Summary:\s*(.*?)(?=\nData Quality Report:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["eda_comprehensive_summary"] = summary_match.group(1).strip() if summary_match else "Summary not parsed."
    parsed_output["eda_identified_issues"] = re.findall(r"-\s*Missing Values:\s*(.+)", final_answer_string, re.IGNORECASE) + re.findall(r"-\s*Outliers:\s*(.+)", final_answer_string, re.IGNORECASE)
    parsed_output["eda_fe_suggestions"] = re.findall(r"-\s*FE Suggestion:\s*(.+)", final_answer_string, re.IGNORECASE)
    parsed_output["eda_key_insights"] = re.findall(r"-\s*Insight:\s*(.+)", final_answer_string, re.IGNORECASE)
    
    plot_refs = {}
    plot_matches = re.findall(r"Plot\s*-\s*([^:]+):\s*(\S+)", final_answer_string, re.IGNORECASE)
    for name, ref in plot_matches: plot_refs[name.strip().lower().replace(" ", "_")] = ref.strip("'\"")
    parsed_output["eda_plot_references"] = plot_refs

    for key, pattern_str in {
        "eda_processed_train_ref": r"Processed Train Data:\s*(\S+)",
        "eda_processed_val_ref": r"Processed Val Data:\s*(\S+)",
        "eda_processed_test_ref": r"Processed Test Data:\s*(\S+)",
    }.items():
        match = re.search(pattern_str, final_answer_string, re.IGNORECASE)
        parsed_output[key] = match.group(1).strip("'\"") if match else f"default_{key}_not_found.pkl"
    return parsed_output

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running ---")
    train_ref_eda = state.get("eda_processed_train_ref", "train_eda.pkl")
    val_ref_eda = state.get("eda_processed_val_ref", "val_eda.pkl")
    test_ref_eda = state.get("eda_processed_test_ref", "test_eda.pkl")
    suggestions = state.get("eda_fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    fe_tool_context_hint = (f"Input data refs from EDA: train='{train_ref_eda}', val='{val_ref_eda}', test='{test_ref_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {suggestions}")

    prompt_content = f"""You are a Feature Engineering Specialist.
    PythonTool takes NL instructions and reports data/object references.
    Context from EDA:
    - Input Train Data Ref: {train_ref_eda} (tool should use this as 'current_train_df')
    - Input Val Data Ref: {val_ref_eda} (as 'current_val_df')
    - Input Test Data Ref: {test_ref_eda} (as 'current_test_df')
    - EDA FE Suggestions: {suggestions if suggestions else 'Perform standard best-practice FE.'}
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct PythonTool to use/load datasets using EDA references.
    2. Based on EDA suggestions, instruct tool to:
        a. Fit transformers (scalers, encoders, imputers) on 'current_train_df'. Ask tool to SAVE each fitted transformer and report its reference (e.g., 'fitted_scaler.pkl').
        b. (Primary Goal) Instruct PythonTool to create a Scikit-learn `Pipeline` object. This pipeline should include all the necessary preprocessing steps (using references to individually fitted transformers OR by defining them directly in the pipeline construction instruction) AND an UNTRAINED model estimator (e.g., RandomForestClassifier). Ask tool to SAVE this untrained full pipeline and report its reference.
    3. As a fallback or for verification, also instruct tool to apply transformations and separate features (X) and target ('{target_col}') from the transformed train/val data, and create X_test. Ask tool to report references for these X_train, y_train, X_val, y_val, X_test and the final feature list.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST include:
    1. FE Applied Steps Summary: <Summary of transformers fitted, pipeline created.>
    2. Final Feature List: (as reported by tool for X datasets, if generated separately)
       - Feature: <feature_name_1>
    3. Transformer References (as reported by PythonTool, if saved individually):
       - Scaler Reference: <tool_reported_scaler_ref>
    4. Untrained Full Pipeline Reference: <tool_reported_untrained_pipeline_ref>
    5. (Optional) Transformed X/y Data References (if generated separately from pipeline):
       - X_train Transformed Reference: <tool_reported_X_train_ref>
       - y_train Reference: <tool_reported_y_train_ref> 
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 12), fe_tool_context_hint)

    parsed_output = {"current_stage_completed": "FeatureEngineering"}
    summary_match = re.search(r"FE Applied Steps Summary:\s*(.*?)(?=\nFinal Feature List:|\nTransformer References:|\nUntrained Full Pipeline Reference:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["fe_applied_steps_summary"] = summary_match.group(1).strip() if summary_match else "FE Summary not parsed."
    parsed_output["fe_final_feature_list"] = re.findall(r"-\s*Feature:\s*(.+)", final_answer_string, re.IGNORECASE)
    
    transformer_refs = {}
    tf_matches = re.findall(r"-\s*(\w+)\s*Reference:\s*(\S+)", final_answer_string, re.IGNORECASE) # Generic transformer parsing
    for tf_name, tf_ref in tf_matches:
        if "pipeline" not in tf_name.lower() and all(kw not in tf_name.lower() for kw in ["x_train", "y_train", "x_val", "y_val", "x_test"]):
            transformer_refs[tf_name.strip().lower()] = tf_ref.strip("'\"")
    parsed_output["fe_transformer_references"] = transformer_refs

    pipeline_ref_match = re.search(r"Untrained Full Pipeline Reference:\s*(\S+)", final_answer_string, re.IGNORECASE)
    parsed_output["fe_untrained_full_pipeline_ref"] = pipeline_ref_match.group(1).strip("'\"") if pipeline_ref_match else "untrained_pipeline_not_parsed.pkl"

    for key, pattern_str in {
        "fe_X_train_transformed_ref": r"X_train Transformed Reference:\s*(\S+)",
        "fe_y_train_ref": r"y_train Reference:\s*(\S+)",
        "fe_X_val_transformed_ref": r"X_val Transformed Reference:\s*(\S+)",
        "fe_y_val_ref": r"y_val Reference:\s*(\S+)",
        "fe_X_test_transformed_ref": r"X_test Transformed Reference:\s*(\S+)",
    }.items():
        match = re.search(pattern_str, final_answer_string, re.IGNORECASE)
        parsed_output[key] = match.group(1).strip("'\"") if match else f"ref_not_parsed_for_{key.replace('fe_','').replace('_transformed_ref','').replace('_ref','')}.pkl"
    parsed_output["fe_y_test_ref"] = None 
    return parsed_output

def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running ---")
    untrained_pipeline_ref = state.get("fe_untrained_full_pipeline_ref", "untrained_pipeline.pkl")
    # Use X/y refs from FE if the pipeline expects already separated data for fitting.
    # If the untrained_pipeline_ref is designed to take data that still includes the target,
    # then it would use eda_processed_train_ref and target_column_name.
    # For this example, assume FE provides X/y that are ready for the *model estimator* part,
    # OR the untrained_pipeline_ref is a full sklearn pipeline (preproc + model estimator)
    # that will be fit on X_train and y_train.
    x_train_ref = state.get("fe_X_train_transformed_ref", "X_train_fe.pkl") 
    y_train_ref = state.get("fe_y_train_ref", "y_train_fe.pkl")
    
    model_tool_context_hint = (f"Untrained pipeline ref: '{untrained_pipeline_ref}'. "
                               f"Train with X_train_ref: '{x_train_ref}', y_train_ref: '{y_train_ref}'.")

    prompt_content = f"""You are a Modeling Specialist. PythonTool takes NL instructions.
    Context from Feature Engineering:
    - Untrained Full Scikit-learn Pipeline Reference: {untrained_pipeline_ref} 
    - X_train Reference (for training the pipeline): {x_train_ref}
    - y_train Reference (for training the pipeline): {y_train_ref}

    Your task:
    1. Instruct PythonTool to load the untrained pipeline from '{untrained_pipeline_ref}'.
    2. Instruct PythonTool to load X_train from '{x_train_ref}' and y_train from '{y_train_ref}'.
    3. Instruct PythonTool to train (fit) the loaded pipeline using this X_train and y_train.
    4. Instruct PythonTool to save the ENTIRE TRAINED PIPELINE to a new file and report its reference.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST include:
    1. Model Training Summary: <Summary of training process.>
    2. Trained Pipeline Reference: <tool_reported_TRAINED_pipeline_ref>
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), model_tool_context_hint)

    parsed_output = {"current_stage_completed": "Modeling"}
    summary_match = re.search(r"Model Training Summary:\s*(.*?)(?=\nT
