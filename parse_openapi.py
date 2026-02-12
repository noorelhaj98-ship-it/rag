import requests, json

base = "http://192.168.2.30:6001"
spec = requests.get(base + "/openapi.json", timeout=10).json()

print("TITLE:", spec.get("info", {}).get("title"))
print("VERSION:", spec.get("info", {}).get("version"))
print("\nPOST endpoints:")

paths = spec.get("paths", {})
post_paths = []
for p, methods in paths.items():
    if "post" in methods:
        post_paths.append((p, methods["post"]))

for p, post in post_paths:
    summary = post.get("summary") or ""
    print(f"  {p}  |  {summary}")

print("\nSchemas:")
schemas = spec.get("components", {}).get("schemas", {})
for name, schema in schemas.items():
    props = schema.get("properties", {})
    required = schema.get("required", [])
    print(f"\n- {name}")
    print("  required:", required)
    print("  properties:", list(props.keys()))
