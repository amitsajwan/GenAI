from openai import OpenAI  # Ensure you have the correct OpenAI client

def ask_user_choice(available_files):
    """
    Uses LLM to ask the user which OpenAPI YAML file to run.
    """
    prompt = f"Available OpenAPI specs: {', '.join(available_files)}. Which one do you want to run?"
    
    # Use OpenAI's model to generate a response
    response = OpenAI().chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are helping select an API specification file."},
                  {"role": "user", "content": prompt}]
    )

    # Extract the selected file name from LLM response
    selected_file = response.choices[0].message.content.strip()

    # Validate if the selection is in the available files list
    if selected_file not in available_files:
        print(f"Invalid selection: {selected_file}. Defaulting to the first file.")
        return available_files[0]
    
    return selected_file
