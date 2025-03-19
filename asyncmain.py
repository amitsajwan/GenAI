import asyncio
from langraph.graph import Graph
from utils.llm_utils import ask_llm_execution_sequence
from utils.api_utils import extract_api_details
from execution.workflow_runner import execute_workflow

async def main():
    """
    Main entry point for the Langraph-based API testing framework.
    - Extracts APIs from an OpenAPI (Swagger) YAML file.
    - Uses LLM to suggest execution sequence.
    - Allows user intervention to modify the sequence.
    - Executes APIs using Langraph.
    """
    print("📂 Reading available OpenAPI specifications...")

    # Extract API details from OpenAPI YAML
    api_details = extract_api_details()

    # Ask LLM to suggest execution sequence
    suggested_sequence = await ask_llm_execution_sequence(api_details)

    print(f"🤖 LLM Suggested Execution Sequence: {suggested_sequence}")

    # Confirm with user
    user_confirmed_sequence = input("Do you want to modify the sequence? (yes/no): ").strip().lower()
    
    if user_confirmed_sequence == "yes":
        new_sequence = input("Enter modified sequence as comma-separated APIs: ").strip().split(",")
        execution_sequence = [api.strip() for api in new_sequence]
    else:
        execution_sequence = suggested_sequence

    print(f"✅ Final Execution Sequence: {execution_sequence}")

    # Execute APIs using Langraph workflow
    await execute_workflow(api_details, execution_sequence)

if __name__ == "__main__":
    asyncio.run(main())  # Run the async function
