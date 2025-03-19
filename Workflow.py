from langraph.graph import Graph

def build_workflow(api_details):
    """
    Constructs a Langraph workflow for executing APIs with user intervention.
    """
    workflow = Graph()

    # Start node
    start = workflow.add_node("start")

    # Create nodes for each API operation
    nodes = {}
    for api_name in api_details.keys():
        nodes[api_name] = workflow.add_node(api_name)

    # User intervention node
    user_verification = workflow.add_node("user_verification")

    # Connect start to user intervention
    workflow.add_transition(start, user_verification)

    # User verifies execution sequence
    for api_name in api_details.keys():
        workflow.add_transition(user_verification, nodes[api_name])

    # End node
    end = workflow.add_node("end")
    for node in nodes.values():
        workflow.add_transition(node, end)

    return workflow
