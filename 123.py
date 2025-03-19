from langraph.graph import Graph

def start_action(state):
    """Initial state action before execution begins."""
    print("Workflow started.")
    return state

def user_verification(state):
    """User verifies and modifies the execution sequence."""
    from user_intervention import refine_execution_sequence
    state["execution_sequence"] = refine_execution_sequence(state["execution_sequence"])
    return state

def execute_api(state, api_name):
    """Executes an individual API call."""
    from execution_flow import run_api_call
    run_api_call(api_name, state)
    return state

def end_action(state):
    """Marks the end of the workflow."""
    print("Workflow completed.")
    return state

def build_workflow(api_details):
    """
    Constructs a Langraph workflow for executing APIs with user intervention.
    """
    workflow = Graph()

    # Start node with a function
    start = workflow.add_node(start_action)

    # User verification node
    verification = workflow.add_node(user_verification)

    # API execution nodes (ensuring uniqueness)
    api_nodes = {}
    for api in api_details.keys():
        def api_execution(state, api_name=api):
            return execute_api(state, api_name)

        api_nodes[api] = workflow.add_node(api_execution)

    # End node using a named function
    end = workflow.add_node(end_action)

    # Define transitions
    workflow.add_transition(start, verification)
    for api, node in api_nodes.items():
        workflow.add_transition(verification, node)
        workflow.add_transition(node, end)  # Each API step leads to the end after execution

    return workflow
