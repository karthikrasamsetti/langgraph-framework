import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import parse_api_keys
import requests

BASE     = "http://localhost:8000"
TEST_KEY = "sk-test-0d9420bfc59b7aa1a6eeb8c5ae3db9c3cb0dd20a8c713531"

print("\n── Parsed API keys ──")
keys = parse_api_keys()
for k, v in keys.items():
    print(f"  {k[:24]}... name={v['name']}, rpm={v['rpm']}")

print("\n── Test 1: No key → expect 401 ──")
r = requests.post(f"{BASE}/chat",
    json={"message": "hello"})
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")

print("\n── Test 2: Wrong key → expect 401 ──")
r = requests.post(f"{BASE}/chat",
    headers={"X-API-Key": "wrong-key"},
    json={"message": "hello"})
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")

print("\n── Test 3: Valid key → expect 200 ──")
r = requests.get(f"{BASE}/api/auth/validate",
    headers={"X-API-Key": TEST_KEY})
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")

print("\n── Test 4: Health (no key) → expect 200 ──")
r = requests.get(f"{BASE}/health")
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")

print("\n── Test 5: Rate limit (6 requests, limit=5) ──")
for i in range(1, 7):
    r = requests.get(f"{BASE}/api/auth/validate",
        headers={"X-API-Key": TEST_KEY})
    limited = "← RATE LIMITED" if r.status_code == 429 else ""
    print(f"  Request {i}: {r.status_code} {limited}")
    if r.status_code == 429:
        print(f"  Detail: {r.json()['detail']}")

print("\nAll tests done.")