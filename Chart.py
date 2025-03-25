import dash
from dash import html, Output, Input, ctx
import dash_cytoscape as cyto
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
data = json.loads(api_dag_json)

# Convert JSON to Cytoscape format
elements = [{"data": {"id": node, "label": node}} for node in data["nodes"]]
elements += [{"data": {"source": edge["from"], "target": edge["to"]}} for edge in data["edges"]]

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Interactive API Execution DAG"),
    
    cyto.Cytoscape(
        id="dag",
        layout={"name": "cose"},
        style={"width": "100%", "height": "500px"},
        elements=elements,
        stylesheet=[
            {"selector": "node", "style": {"content": "data(label)", "background-color": "lightblue", "font-size": "15px"}},
            {"selector": "edge", "style": {"line-color": "gray", "width": 2}},
        ],
    ),
    
    html.Button("Save Execution Order", id="save-btn", n_clicks=0),
    html.Button("Delete Selected", id="delete-btn", n_clicks=0, style={"margin-left": "10px"}),
    html.Pre(id="output", style={"border": "1px solid black", "padding": "10px", "margin-top": "10px"}),
])

@app.callback(
    Output("dag", "elements"),
    Input("delete-btn", "n_clicks"),
    Input("dag", "elements"),
    prevent_initial_call=True
)
def delete_selected_edge(n_clicks, elements):
    """Delete the selected node/edge when 'Delete Selected' is clicked."""
    if ctx.triggered_id == "delete-btn":
        return [e for e in elements if not e.get("selected", False)]
    return elements

@app.callback(
    Output("output", "children"),
    Input("save-btn", "n_clicks"),
    Input("dag", "elements")
)
def save_execution_order(n_clicks, elements):
    """Save the new execution order after modifying edges."""
    if n_clicks > 0:
        new_edges = [
            {"from": edge["data"]["source"], "to": edge["data"]["target"]}
            for edge in elements if "source" in edge["data"]
        ]
        new_nodes = list(set(node["data"]["id"] for node in elements if "id" in node["data"]))
        updated_json = json.dumps({"nodes": new_nodes, "edges": new_edges}, indent=2)
        return updated_json
    return "Modify the graph and click 'Save Execution Order'."

if __name__ == "__main__":
    app.run_server(debug=True)
  
