import dash
from dash import dcc, html, Input, Output
import dash_cytoscape as cyto
import networkx as nx
import json

# API Execution DAG JSON
api_dag_json = '''
{
  "nodes": ["/users", "/users/{id}", "/orders", "/orders/{id}", "/payments"],
  "edges": [
    {"from": "/users", "to": "/users/{id}"},
    {"from": "/users/{id}", "to": "/orders"},
    {"from": "/orders", "to": "/orders/{id}"},
    {"from": "/orders/{id}", "to": "/payments"}
  ]
}
'''

# Load JSON
data = json.loads(api_dag_json)

# Create DAG
G = nx.DiGraph()
G.add_nodes_from(data["nodes"])
G.add_edges_from([(edge["from"], edge["to"]) for edge in data["edges"]])

# Convert DAG to Cytoscape format
elements = [{"data": {"id": node, "label": node}} for node in G.nodes()]
elements += [{"data": {"source": edge["from"], "target": edge["to"]}} for edge in data["edges"]]

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Interactive API Execution DAG"),
    
    cyto.Cytoscape(
        id="dag",
        layout={"name": "cose"},  # Auto-layout
        style={"width": "100%", "height": "500px"},
        elements=elements,
        stylesheet=[
            {"selector": "node", "style": {"content": "data(label)", "background-color": "lightblue", "font-size": "15px"}},
            {"selector": "edge", "style": {"line-color": "gray", "width": 2}},
        ],
    ),

    html.Button("Save Execution Order", id="save-btn", n_clicks=0),
    html.Pre(id="output", style={"border": "1px solid black", "padding": "10px", "margin-top": "10px"}),
])

@app.callback(
    Output("output", "children"),
    Input("save-btn", "n_clicks"),
    Input("dag", "elements")
)
def update_execution_order(n_clicks, elements):
    if n_clicks > 0:
        # Extract nodes and edges from modified graph
        new_edges = [
            {"from": edge["data"]["source"], "to": edge["data"]["target"]}
            for edge in elements if "source" in edge["data"]
        ]
        new_nodes = list(set(node["data"]["id"] for node in elements if "id" in node["data"]))

        # Update JSON dynamically
        updated_json = json.dumps({"nodes": new_nodes, "edges": new_edges}, indent=2)
        return updated_json
    return "Click 'Save Execution Order' to update JSON."

if __name__ == "__main__":
    app.run_server(debug=True)
