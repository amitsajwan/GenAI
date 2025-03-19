import os
from api_extractor import extract_api_details
from execution_flow import execute_workflow
from user_intervention import refine_execution_sequence
from load_testing import run_load_test
from llm_utils import ask_user_choice  # Utility for LLM-based user interaction

OPENAPI_FOLDER = "openapi_specs"

def list_openapi_files():
    """Lists available OpenAPI YAML files in the specified folder."""
    return [f for f in os.listdir(OPENAPI_FOLDER) if f.endswith((".yaml", ".yml"))]

def main():
    # Step 1: List available OpenAPI files
    available_files = list_openapi_files()
    if not available_files:
        print("No OpenAPI files found.")
        return

    # Step 2: Ask user which Swagger file to run (using LLM)
    selected_file = ask_user_choice(available_files)

    # Step 3: Extract API details from the chosen file
    api_details = extract_api_details(os.path.join(OPENAPI_FOLDER, selected_file))

    # Step 4: Generate execution sequence dynamically
    suggested_sequence = list(api_details.keys())  # Extracted API operation IDs

    # Step 5: Allow user to refine execution sequence
    final_sequence = refine_execution_sequence(suggested_sequence)

    # Step 6: Execute the workflow
    execute_workflow(final_sequence)

    # Step 7: Perform load testing if required
    run_load_test(final_sequence, users=10)

if __name__ == "__main__":
    main()
