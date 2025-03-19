from api_extractor import extract_api_details
from execution_flow import execute_workflow
from user_intervention import refine_execution_sequence
from load_testing import run_load_test

def main():
    # Step 1: Extract API details
    api_details = extract_api_details("openapi_spec.yaml")

    # Step 2: Generate execution sequence
    suggested_sequence = ["Create User", "Get User Details", "Update User Role", "List All Users"]

    # Step 3: User refines execution sequence
    final_sequence = refine_execution_sequence(suggested_sequence)

    # Step 4: Execute the workflow with user validation
    execute_workflow(final_sequence)

    # Step 5: Perform load testing if required
    run_load_test(final_sequence, users=10)

if __name__ == "__main__":
    main()
