"""
Test script to call Ollama API directly to diagnose timeout issue.
"""
import urllib.request
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings

url = f"{settings.OLLAMA_URL}/api/chat"
model = settings.OLLAMA_MODEL

print("="*70)
print("Direct Ollama API Test")
print("="*70)
print(f"URL: {url}")
print(f"Model: {model}")
print("="*70)

body = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, respond with 'Ollama is working'"},
    ],
    "options": {"temperature": 0.2, "num_ctx": 768},
    "stream": False,
}).encode("utf-8")

request = urllib.request.Request(
    url, 
    data=body, 
    headers={"Content-Type": "application/json"}, 
    method="POST"
)

try:
    print("Sending request to Ollama API...")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    
    print("✅ Ollama API call successful!")
    print(f"Response: {payload['message']['content']}")
    print("="*70)
    
except urllib.error.HTTPError as http_err:
    print(f"❌ HTTP Error: {http_err.code}")
    try:
        err_body = http_err.read().decode("utf-8")
        print(f"Error body: {err_body[:500]}")
    except:
        pass
    print("="*70)
    
except urllib.error.URLError as url_err:
    print(f"❌ Network Error: {url_err}")
    print("="*70)
    
except Exception as exc:
    print(f"❌ Unexpected Error: {exc}")
    import traceback
    traceback.print_exc()
    print("="*70)
