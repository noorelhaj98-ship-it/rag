import requests, json

base = "http://192.168.2.30:6001"
spec = requests.get(base + "/openapi.json", timeout=10).json()

embed_post = spec["paths"]["/embed"]["post"]
print("SUMMARY:", embed_post.get("summary"))

# Request body schema
rb = embed_post.get("requestBody", {})
print("\nrequestBody keys:", rb.keys())

content = rb.get("content", {})
print("\ncontent types:", list(content.keys()))

for ctype, obj in content.items():
    schema = obj.get("schema", {})
    print("\n--- schema for", ctype, "---")
    print(json.dumps(schema, indent=2))

# Also print component schemas (if used via $ref)
schemas = spec.get("components", {}).get("schemas", {})
if schemas:
    print("\n--- components.schemas ---")
    for name, s in schemas.items():
        print("\n#", name)
        print(json.dumps(s, indent=2))
