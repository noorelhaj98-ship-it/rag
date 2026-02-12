import requests

base = "http://192.168.2.30:6001"

paths = ["/", "/health", "/docs", "/openapi.json", "/embed", "/embeddings", "/v1/embeddings"]

for p in paths:
    try:
        r = requests.get(base + p, timeout=10)
        print(f"{p:15} -> {r.status_code}  {r.headers.get('content-type')}")
        if p == "/openapi.json" and r.status_code == 200:
            print("openapi.json length:", len(r.text))
    except Exception as e:
        print(f"{p:15} -> ERROR {e}")
