#!/usr/bin/env python
"""Test script to verify RBAC is working."""
import os
import sys

# Set API keys before importing server
os.environ['API_KEYS'] = 'alice:alice123:user,bob:bob456:admin,guest:guest789:readonly'
os.environ['RATE_LIMIT'] = '100/minute'

# Now import server
from server import RBAC_ENABLED, API_KEYS, ROLE_PERMISSIONS, verify_token, require_permission
from fastapi.security import HTTPAuthorizationCredentials

def test_rbac():
    print("=" * 50)
    print("RBAC TEST RESULTS")
    print("=" * 50)
    
    # Test 1: Check RBAC is enabled
    print(f"\n1. RBAC Enabled: {RBAC_ENABLED}")
    assert RBAC_ENABLED == True, "RBAC should be enabled!"
    print("   ✓ PASS")
    
    # Test 2: Check users loaded
    print(f"\n2. Configured Users: {list(API_KEYS.keys())}")
    assert len(API_KEYS) == 3, "Should have 3 users!"
    print("   ✓ PASS")
    
    # Test 3: Check role permissions
    print(f"\n3. Role Permissions:")
    for role, perms in ROLE_PERMISSIONS.items():
        print(f"   - {role.value}: {perms}")
    print("   ✓ PASS")
    
    # Test 4: Test user details
    print(f"\n4. User Details:")
    for key, info in API_KEYS.items():
        print(f"   - {info['name']} (key: {key}): role={info['role'].value}")
    print("   ✓ PASS")
    
    # Test 5: Verify token function
    print(f"\n5. Testing Token Verification:")
    
    # Test valid token
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="alice123")
    user = verify_token(creds)
    print(f"   - Valid token 'alice123': user={user['name']}, role={user['role'].value}")
    assert user['name'] == 'alice'
    assert user['role'].value == 'user'
    
    # Test invalid token
    try:
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid")
        verify_token(creds)
        print("   ✗ FAIL: Should have raised exception for invalid token")
        sys.exit(1)
    except Exception as e:
        print(f"   - Invalid token correctly rejected: {type(e).__name__}")
    
    # Test no token
    try:
        verify_token(None)
        print("   ✗ FAIL: Should have raised exception for missing token")
        sys.exit(1)
    except Exception as e:
        print(f"   - Missing token correctly rejected: {type(e).__name__}")
    
    print("   ✓ PASS")
    
    # Test 6: Test permissions
    print(f"\n6. Testing Permission Checks:")
    
    # User should have 'ask' permission
    checker = require_permission("ask")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="alice123")
    user = verify_token(creds)
    result = checker(user)
    print(f"   - User 'alice' has 'ask' permission: ✓")
    
    # Readonly should NOT have 'ask' permission
    checker = require_permission("ask")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="guest789")
    user = verify_token(creds)
    try:
        checker(user)
        print("   ✗ FAIL: Readonly user should not have 'ask' permission")
        sys.exit(1)
    except Exception as e:
        print(f"   - Readonly user correctly denied 'ask' permission: ✓")
    
    print("   ✓ PASS")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nRBAC is properly configured and working.")
    print("\nTo use with the server:")
    print("1. Stop the current server (Ctrl+C)")
    print("2. Start fresh with: python -m uvicorn server:app --host 0.0.0.0 --port 8000")
    print("3. Or set env var explicitly:")
    print("   $env:API_KEYS='alice:alice123:user,bob:bob456:admin,guest:guest789:readonly'")
    print("   python -m uvicorn server:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    test_rbac()
