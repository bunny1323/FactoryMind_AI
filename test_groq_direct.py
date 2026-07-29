"""
Test script to call Groq API directly to diagnose the issue.
"""
import urllib.request
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings

api_key = settings.GROQ_API_KEY
model = settings.GROQ_MODEL

print("="*70)
print("Direct Groq API Test")
print("="*70)
print(f"API Key: {api_key[:12]}...")
print(f"Model: {model}")
print("="*70)

url = "https://api.groq.com/openai/v1/chat/completions"
body = json.dumps({
    "model": model,
    "temperature": 0.15,
    "max_tokens": 1024,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you respond with 'Groq is working'?"},
    ],
}).encode("utf-8")

request = urllib.request.Request(
    url,
    data=body,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    print("Sending request to Groq API...")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    
    print("✅ Groq API call successful!")
    print(f"Response: {payload['choices'][0]['message']['content']}")
    print("="*70)
    
except urllib.error.HTTPError as http_err:
    print(f"❌ HTTP Error: {http_err.code}")
    try:
        err_body = http_err.read().decode("utf-8")
        err_json = json.loads(err_body)
        print(f"Error details: {err_json}")
    except:
        print(f"Error body: {err_body[:500]}")
    print("="*70)
    
except urllib.error.URLError as url_err:
    print(f"❌ Network Error: {url_err}")
    print("="*70)
    
except Exception as exc:
    print(f"❌ Unexpected Error: {exc}")
    import traceback
    traceback.print_exc()
    print("="*70)
