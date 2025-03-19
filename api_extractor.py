import yaml

def extract_api_details(yaml_file):
    with open(yaml_file, 'r') as file:
        openapi_spec = yaml.safe_load(file)
    return openapi_spec  # Return parsed OpenAPI data
