import io
import time
import matplotlib.pyplot as plt
import httpx  # Use async requests
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics at /metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/chart")
async def chart():
    """Generate a real-time API latency chart and return as an image."""
    metrics_url = "http://localhost:8000/metrics"

    async with httpx.AsyncClient() as client:
        response = await client.get(metrics_url)

    if response.status_code != 200:
        return {"error": "Failed to fetch metrics"}

    latencies = []
    api_names = []

    for line in response.text.split("\n"):
        if line.startswith("api_latency_seconds_bucket") and 'le="0.5"' in line:
            parts = line.split()
            labels = parts[0].split("{")[-1].split("}")[0].split(",")
            api_name = labels[0].split("=")[-1].strip('"')
            latency = float(parts[-1])
            latencies.append(latency)
            api_names.append(api_name)

    if not latencies:
        return {"error": "No latency data available"}

    # Generate a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(api_names, latencies, color="blue")
    plt.xlabel("API Name")
    plt.ylabel("Latency (seconds)")
    plt.title("API Latency Chart")
    plt.xticks(rotation=45)

    # Convert plot to image in memory
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    plt.close()

    return Response(content=img_buf.getvalue(), media_type="image/png")
