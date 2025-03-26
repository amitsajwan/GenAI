import yaml
import requests
import time
import random

class OpenAPIExecutor:
    def __init__(self, spec_path, base_url, headers=None):
        self.spec_path = spec_path
        self.base_url = base_url
        self.headers = headers if headers else {}
        self.api_list = self.parse_openapi_spec()
        self.response_cache = {}  # Stores API responses dynamically

    def parse_openapi_spec(self):
        """Parses OpenAPI YAML and extracts API details."""
        with open(self.spec_path, "r") as file:
            spec = yaml.safe_load(file)

        api_endpoints = []
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                operation_id = details.get("operationId", f"{method}_{path.replace('/', '_')}")
                request_body = details.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
                parameters = details.get("parameters", [])
                responses = details.get("responses", {})

                api_endpoints.append({
                    "method": method.upper(),
                    "endpoint": path,
                    "operation_id": operation_id,
                    "request_body": request_body,
                    "parameters": parameters,
                    "responses": responses
                })
        return api_endpoints


    def generate_example(self, schema): # Added self as the first argument
        """Generate example data from an OpenAPI schema definition."""
        if "type" in schema:
            if schema["type"] == "string":
                return "John Doe"
            elif schema["type"] == "integer":
                return random.randint(1, 1000)
            elif schema["type"] == "boolean":
                return random.choice([True, False])
            elif schema["type"] == "array":
                return [self.generate_example(schema["items"])] # Call using self
            elif schema["type"] == "object":
                return {k: self.generate_example(v) for k, v in schema.get("properties", {}).items()} # Call using self
        return None  # Default case

    def resolve_payload(self, payload_schema):
        """Resolves OpenAPI schema to actual data."""
        return self.generate_example(payload_schema)  # Call using self

    # def resolve_payload(self, payload):
    #     """Replaces placeholders in request payload with values from previous responses."""
    #     if isinstance(payload, dict):
    #         for key, value in payload.items():
    #             if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
    #                 ref_key = value.strip("{}")
    #                 payload[key] = self.response_cache.get(ref_key, value)
    #     return payload

    def execute_api(self, api_info):
        """Executes an API call and stores relevant response data."""
        method = api_info["method"]
        url = f"{self.base_url}{api_info['endpoint']}"

        # Resolve request body dependencies
        payload = self.resolve_payload(api_info.get("request_body", {}).get("properties", {}))

        print(f"Executing: {method} {url} with payload {payload}")
        # response = requests.request(method, url, headers=self.headers, json=payload)

        # print(f"Response [{response.status_code}]: {response.text}")

        # if response.status_code in [200, 201]:
        #     try:
        #         self.response_cache.update(response.json())  # Store API response for dependencies
        #     except Exception as e:
        #         print(f"Error parsing JSON response: {e}")

        return "response"

    def run_sequence(self):
        """Executes APIs sequentially, resolving dependencies dynamically."""
        for api in self.api_list:
            self.execute_api(api)
            time.sleep(1)  # Prevent hitting rate limits

# Usage Example
executor = OpenAPIExecutor(
    spec_path="sample_data/openapi.yaml",
    base_url="https://api.example.com",
    headers={"Authorization": "Bearer your_token"}
)
executor.run_sequence()
