from pydantic import BaseModel
from typing import Optional, Dict, List
import time

class APIExecutionState(BaseModel):
    """
    Defines execution state for API testing workflow, including metrics.
    """
    last_api: Optional[str] = None  # Last executed API
    next_api: Optional[str] = None  # Next API to execute
    execution_results: Dict[str, Dict] = {}  # Stores API responses & status codes
    latency: List[float] = []  # Stores response time per request
    status_count: Dict[str, int] = {"success": 0, "failure": 0}  # Success/Failure count

    def record_metrics(self, api_name: str, latency: float, success: bool):
        """Updates state with execution metrics."""
        self.last_api = api_name
        self.latency.append(latency)
        if success:
            self.status_count["success"] += 1
        else:
            self.status_count["failure"] += 1

from prometheus_client import Counter, Histogram, start_http_server
import asyncio

# Define Prometheus Metrics
API_CALLS = Counter("api_calls_total", "Total API calls", ["status"])
API_LATENCY = Histogram("api_latency_seconds", "API response time", ["api_name"])

async def execute_test(chain, state):
    """
    Executes a single API test, records metrics, and updates the state.
    """
    start_time = time.time()
    try:
        async for result in chain.astream(state, stream_mode="values"):
            state.execution_results.update(result)

        API_CALLS.labels(status="success").inc()
        success = True
    except Exception:
        API_CALLS.labels(status="failure").inc()
        success = False

    latency = time.time() - start_time
    API_LATENCY.labels(api_name=state.last_api or "unknown").observe(latency)
    
    # Store metrics in state for reporting
    state.record_metrics(state.last_api or "unknown", latency, success)

async def run_load_test():
    """
    Runs a load test with 100 concurrent users, tracks metrics, and updates state.
    """
    chain = langgraph.compile()  # Compile LangGraph workflow
    states = [APIExecutionState() for _ in range(100)]  # Create 100 user states

    start_http_server(8000)  # Expose metrics at http://localhost:8000/metrics

    # Run all executions concurrently
    await asyncio.gather(*[execute_test(chain, state) for state in states])

    # Generate performance report
    generate_report(states)


def generate_report(states):
    """
    Generates a performance report from API execution state metrics.
    """
    total_calls = sum(s.status_count["success"] + s.status_count["failure"] for s in states)
    total_success = sum(s.status_count["success"] for s in states)
    total_failure = sum(s.status_count["failure"] for s in states)
    avg_latency = sum(sum(s.latency) for s in states) / total_calls if total_calls else 0

    print("\n=== Load Test Report ===")
    print(f"Total API Calls: {total_calls}")
    print(f"Successful Calls: {total_success}")
    print(f"Failed Calls: {total_failure}")
    print(f"Average Latency: {avg_latency:.4f} sec")
