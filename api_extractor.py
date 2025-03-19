import yaml

def extract_api_details(yaml_file):
    with open(yaml_file, 'r') as file:
        openapi_spec = yaml.safe_load(file)

    apis = {}
    for path, methods in openapi_spec.get("paths", {}).items():
        for method, details in methods.items():
            operation_id = details.get("operationId", f"{method.upper()} {path}")
            request_body = details.get("requestBody", {})
            responses = details.get("responses", {})
            
            # Extract schema if available
            schema = None
            if "content" in request_body:
                for content_type, content_details in request_body["content"].items():
                    schema = content_details.get("schema")
                    break

            apis[operation_id] = {
                "method": method.upper(),
                "path": path,
                "schema": schema,
                "responses": responses
            }
    
    return apis  # Returns extracted API details with payload schema
