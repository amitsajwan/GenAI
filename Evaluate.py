import re
import json # For context hint formatting, though not strictly used by this node's output directly
from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # Needed for run_generic_react_loop
from langchain_openai import ChatOpenAI # Needed for run_generic_react_loop

# Assume MultiAgentPipelineState is defined as in the full pipeline:
class MultiAgentPipelineState(TypedDict):
    # ... (other fields from the full state definition)
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 
    problem_type: Optional[Literal["classification", "regression"]] 

    fe_X_val_transformed_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_transformed_ref: Optional[str]  
    
    model_trained_pipeline_ref: Optional[str] 

    evaluation_summary: Optional[str] 
    evaluation_metrics: Optional[Dict[str, float]]
    test_set_prediction_status: Optional[str]

    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]

# Assume llm is initialized globally or passed appropriately for run_generic_react_loop
# llm = ChatOpenAI(model="gpt-4o", temperature=0.1) 

# Assume agno_python_tool_interface is defined as in the full pipeline:
# def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
#     # ... your tool implementation or simulation ...
#     print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction:\n    '{instruction}'")
#     sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
#     if "load the trained pipeline" in instruction and "load x_val" in instruction:
#         sim_observation += "Trained pipeline and validation data loaded successfully."
#     elif "make predictions on x_val" in instruction:
#         sim_observation += "Predictions made on X_val. Predictions stored as 'val_predictions'."
#     elif "calculate classification metrics" in instruction:
#         sim_observation += "Metrics calculated: Accuracy: 0.91, F1-Score: 0.89, ROC AUC: 0.95."
#     elif "calculate regression metrics" in instruction:
#         sim_observation += "Metrics calculated: MSE: 15.2, R-squared: 0.78."
#     elif "make predictions on x_test" in instruction:
#         sim_observation += "Predictions made on X_test. Saved to 'test_predictions_output.csv'."
#     else:
#         sim_observation += "Evaluation task completed."
#     return sim_observation


# Assume run_generic_react_loop is defined as in the full pipeline:
# def run_generic_react_loop(
#     initial_prompt_content: str,
#     max_iterations: int,
#     agent_context_hint_for_tool: Optional[str] = None 
# ) -> str: 
#     # ... implementation of the ReAct loop ...
#     react_messages: List[BaseMessage] = [SystemMessage(content=initial_prompt_content)]
#     final_answer_text = "Agent did not produce a Final Answer within iteration limit."
#     for i in range(max_iterations):
#         # Simplified loop for brevity in this standalone example
#         if i == 0: # First call to LLM
#             # In a real scenario, this would call llm.invoke(react_messages)
#             # and then parse for Action or Final Answer.
#             # For this standalone function, we'll assume the LLM directly gives a final answer
#             # based on a well-crafted prompt.
#             # This is a major simplification of the ReAct loop for showing just the eval node.
#             # The full pipeline code has the actual loop.
#             if "classification" in initial_prompt_content:
#                  final_answer_text = """Final Answer:
# Evaluation Summary: Model evaluation completed on validation set. Performance is good. Predictions also made on test set.
# Validation Metrics:
# - Accuracy: 0.91
# - F1-Score: 0.89
# - ROC AUC: 0.95
# Test Set Prediction Status: Predictions made on test set referenced by 'X_test_final_fe.pkl' and saved to 'test_predictions_output.csv'.
# """
#             else: # Regression
#                  final_answer_text = """Final Answer:
# Evaluation Summary: Model evaluation completed on validation set. Model shows decent predictive power.
# Validation Metrics:
# - MSE: 15.2
# - R-squared: 0.78
# Test Set Prediction Status: Test set not processed as it was not provided or requested for detailed prediction saving in this run.
# """
#             break # Exit loop after one simulated LLM response for this standalone example
#     return final_answer_text


def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("model_trained_pipeline_ref", "trained_model_pipeline.pkl")
    x_val_ref = state.get("fe_X_val_transformed_ref", "X_val_final_fe.pkl")
    y_val_ref = state.get("fe_y_val_ref", "y_val_final_fe.pkl")
    x_test_ref = state.get("fe_X_test_transformed_ref") 
    target_column_name = state.get("target_column_name", "Target")
    problem_type = state.get("problem_type", "classification") # Default to classification

    eval_tool_context_hint = (
        f"Trained pipeline reference: '{trained_pipeline_ref}'. "
        f"Validation X reference: '{x_val_ref}'. Validation y reference: '{y_val_ref}'. "
        f"Test X reference: '{x_test_ref if x_test_ref else 'Not provided'}'. "
        f"Target column name for context: '{target_column_name}'. Problem type: {problem_type}."
    )

    metrics_to_request = "accuracy, precision, recall, F1-score, ROC AUC" if problem_type == "classification" \
        else "MSE, RMSE, MAE, R-squared"

    prompt_content = f"""You are an Evaluation Specialist.
    PythonTool takes NL instructions and reports back results, including metrics.
    Context: {eval_tool_context_hint}

    Your tasks using PythonTool:
    1. Instruct PythonTool to load the trained pipeline from '{trained_pipeline_ref}'.
    2. Instruct PythonTool to load the validation data: X_val from '{x_val_ref}' and y_val from '{y_val_ref}'.
    3. Instruct PythonTool to use the loaded pipeline to make predictions on X_val.
    4. Instruct PythonTool to calculate relevant evaluation metrics: {metrics_to_request} by comparing predictions against y_val. Ask it to report these as key-value pairs.
    5. (Optional) If X_test reference ('{x_test_ref}') is available, instruct PythonTool to make predictions on X_test and report that predictions have been made (e.g., saved to 'test_predictions.csv').

    ReAct Format:
    Action: Python
    Action Input: <NL instruction, e.g., "Load trained pipeline from '{trained_pipeline_ref}'. Then load X_val from '{x_val_ref}' and y_val from '{y_val_ref}'.">
    (System will provide Observation:)
    Observation: <result from PythonTool>

    "Final Answer:" MUST include:
    1. Evaluation Summary: <Brief summary of evaluation process and findings.>
    2. Validation Metrics: (as key-value pairs reported by PythonTool, each on a new line prefixed with '-')
       - Metric Name 1: <value1>
       - Metric Name 2: <value2>
       (e.g., - Accuracy: 0.92)
    3. Test Set Prediction Status: <e.g., "Predictions made on test set referenced by '{x_test_ref}' and saved to 'test_predictions.csv'." or "Test set not processed.">
    Begin.
    """

    # In the full pipeline, this calls the actual run_generic_react_loop
    # For this standalone function, we'd need to define or import it.
    # We'll assume it's available and works as intended.
    final_answer_string = run_generic_react_loop(
        prompt_content,
        state.get("max_react_iterations", 5), 
        eval_tool_context_hint
    )

    parsed_output = {"current_stage_completed": "Evaluation"}
    
    summary_match = re.search(r"Evaluation Summary:\s*(.*?)(?=\nValidation Metrics:|\nTest Set Prediction Status:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["evaluation_summary"] = summary_match.group(1).strip() if summary_match else "Evaluation summary not parsed."

    metrics = {}
    # Regex to find lines like "- Metric Name: 123.45" or "- Metric-Name: 0.9"
    metric_matches = re.findall(r"-\s*([\w\s\-]+):\s*([\d\.]+)", final_answer_string) 
    for name, value in metric_matches:
        try:
            metrics[name.strip()] = float(value)
        except ValueError:
            print(f"Warning: Could not parse metric value as float: {name}={value}")
    parsed_output["evaluation_metrics"] = metrics

    test_status_match = re.search(r"Test Set Prediction Status:\s*(.+)", final_answer_string, re.IGNORECASE)
    parsed_output["test_set_prediction_status"] = test_status_match.group(1).strip() if test_status_match else "Test set status not reported."
    
    return parsed_output
