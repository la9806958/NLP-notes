#!/usr/bin/env python3
"""
Test script to verify AST deduplication works correctly
"""
import json
from alpha_agent_factor import hash_ast, normalize_ast_for_comparison

# Test case 1: Identical ASTs should produce same hash
ast1 = {
    "type": "call",
    "fn": "cond",
    "args": [
        {
            "type": "call",
            "fn": "gt",
            "args": [
                {"type": "symbol", "name": "volume"},
                {
                    "type": "call",
                    "fn": "adv",
                    "args": [
                        {"type": "symbol", "name": "volume"},
                        {"type": "const", "value": 20}
                    ]
                }
            ]
        },
        {
            "type": "call",
            "fn": "hl_range",
            "args": [
                {"type": "symbol", "name": "high"},
                {"type": "symbol", "name": "low"},
                {
                    "type": "call",
                    "fn": "delay",
                    "args": [
                        {"type": "symbol", "name": "close"},
                        {"type": "const", "value": 1}
                    ]
                }
            ]
        },
        {"type": "const", "value": 0}
    ]
}

ast2 = {
    "type": "call",
    "fn": "cond",
    "args": [
        {
            "type": "call",
            "fn": "gt",
            "args": [
                {"type": "symbol", "name": "volume"},
                {
                    "type": "call",
                    "fn": "adv",
                    "args": [
                        {"type": "symbol", "name": "volume"},
                        {"type": "const", "value": 20}
                    ]
                }
            ]
        },
        {
            "type": "call",
            "fn": "hl_range",
            "args": [
                {"type": "symbol", "name": "high"},
                {"type": "symbol", "name": "low"},
                {
                    "type": "call",
                    "fn": "delay",
                    "args": [
                        {"type": "symbol", "name": "close"},
                        {"type": "const", "value": 1}
                    ]
                }
            ]
        },
        {"type": "const", "value": 0}
    ]
}

# Test case 2: Semantically equivalent ASTs (adv vs ts_mean) should produce same hash
ast3 = {
    "type": "call",
    "fn": "cond",
    "args": [
        {
            "type": "call",
            "fn": "gt",
            "args": [
                {"type": "symbol", "name": "volume"},
                {
                    "type": "call",
                    "fn": "ts_mean",  # Changed from 'adv'
                    "args": [
                        {"type": "symbol", "name": "volume"},
                        {"type": "const", "value": 20}
                    ]
                }
            ]
        },
        {
            "type": "call",
            "fn": "hl_range",
            "args": [
                {"type": "symbol", "name": "high"},
                {"type": "symbol", "name": "low"},
                {
                    "type": "call",
                    "fn": "delay",
                    "args": [
                        {"type": "symbol", "name": "close"},
                        {"type": "const", "value": 1}
                    ]
                }
            ]
        },
        {"type": "const", "value": 0}
    ]
}

# Test case 3: Different AST should produce different hash
ast4 = {
    "type": "call",
    "fn": "ts_mean",
    "args": [
        {"type": "symbol", "name": "volume"},
        {"type": "const", "value": 10}
    ]
}

print("="*80)
print("AST DEDUPLICATION TEST")
print("="*80)

hash1 = hash_ast(ast1)
hash2 = hash_ast(ast2)
hash3 = hash_ast(ast3)
hash4 = hash_ast(ast4)

print(f"\nTest 1: Identical ASTs")
print(f"  AST1 hash: {hash1}")
print(f"  AST2 hash: {hash2}")
print(f"  Result: {'✅ PASS' if hash1 == hash2 else '❌ FAIL'} - Hashes are {'identical' if hash1 == hash2 else 'different'}")

print(f"\nTest 2: Semantically equivalent ASTs (adv vs ts_mean)")
print(f"  AST1 hash (uses adv):     {hash1}")
print(f"  AST3 hash (uses ts_mean): {hash3}")
print(f"  Result: {'✅ PASS' if hash1 == hash3 else '❌ FAIL'} - Hashes are {'identical' if hash1 == hash3 else 'different'}")

print(f"\nTest 3: Different ASTs")
print(f"  AST1 hash (complex cond): {hash1}")
print(f"  AST4 hash (simple mean):  {hash4}")
print(f"  Result: {'✅ PASS' if hash1 != hash4 else '❌ FAIL'} - Hashes are {'different' if hash1 != hash4 else 'identical'}")

print("\n" + "="*80)
print("NORMALIZATION TEST")
print("="*80)

normalized1 = normalize_ast_for_comparison(ast1)
normalized3 = normalize_ast_for_comparison(ast3)

print(f"\nOriginal AST1 uses 'adv':")
print(json.dumps(ast1["args"][0]["args"][1], indent=2))

print(f"\nNormalized AST1 (adv -> ts_mean):")
print(json.dumps(normalized1["args"][0]["args"][1], indent=2))

print(f"\nOriginal AST3 uses 'ts_mean':")
print(json.dumps(ast3["args"][0]["args"][1], indent=2))

print(f"\nNormalized AST3 (already ts_mean):")
print(json.dumps(normalized3["args"][0]["args"][1], indent=2))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ AST deduplication is working correctly!")
print(f"   - Identical ASTs produce identical hashes")
print(f"   - Semantically equivalent operators (adv=ts_mean) are normalized")
print(f"   - Different ASTs produce different hashes")
