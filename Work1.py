from llm_utils import get_user_feedback

def refine_execution_sequence(suggested_sequence):
    """
    Allows the user to modify the suggested execution sequence using LLM interaction.
    
    Args:
        suggested_sequence (list): The initial sequence of API operations suggested based on OpenAPI.
    
    Returns:
        list: The final, user-approved execution sequence.
    """
    print("\nSuggested Execution Sequence:")
    for idx, step in enumerate(suggested_sequence, start=1):
        print(f"{idx}. {step}")

    # Ask the user if they want to modify the sequence
    user_modified_sequence = get_user_feedback(suggested_sequence)

    return user_modified_sequence
