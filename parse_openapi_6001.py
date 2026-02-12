import requests

base = "http://192.168.2.30:6001"
spec = requests.get(base + "/openapi.json", timeout=10).json()

print("TITLE:", spec.get("info", {}).get("title"))
print("VERSION:", spec.get("info", {}).get("version"))
print("\nPOST endpoints:")

paths = spec.get("paths", {})
for p, methods in paths.items():
    if "post" in methods:
        post = methods["post"]
        summary = post.get("summary") or ""
        print(f"  {p}  |  {summary}")

print("\nTry these in your .env as EMBEDDING_API_URL:")
for p, methods in paths.items():
    if "post" in methods:
        print(base + p)
