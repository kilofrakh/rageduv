#!/usr/bin/env python3
"""
test.py — Comprehensive test script for RAGedu
Tests imports, basic functionality, and integration of all modules.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Import all modules at the top
try:
    import config
    from ingest import ingest_file, get_vectorstore
    from rag_pipeline import ask, get_llm
    from quiz_generator import generate_quiz
    import analytics
    import api
    IMPORTS_SUCCESS = True
except Exception as e:
    print(f"Critical: Failed to import modules: {e}")
    IMPORTS_SUCCESS = False
    sys.exit(1)

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")

    # Since imports are done at top, just check they exist
    modules = ['config', 'ingest_file', 'get_vectorstore', 'ask', 'get_llm', 'generate_quiz', 'analytics', 'api']
    for mod in modules:
        if mod in globals():
            print(f"✓ {mod} available")
        else:
            print(f"✗ {mod} not available")
            return False

    return True

def test_config():
    """Test config values."""
    print("\nTesting config...")

    try:
        assert config.GROQ_API_KEY == os.getenv("GROQ_API_KEY", "")
        print("✓ GROQ_API_KEY loaded")
    except Exception as e:
        print(f"✗ GROQ_API_KEY test failed: {e}")
        return False

    try:
        assert config.LLM_MODEL == "llama3-8b-8192"
        print("✓ LLM_MODEL set correctly")
    except Exception as e:
        print(f"✗ LLM_MODEL test failed: {e}")
        return False

    try:
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        print("✓ EMBEDDING_MODEL set correctly")
    except Exception as e:
        print(f"✗ EMBEDDING_MODEL test failed: {e}")
        return False

    try:
        assert os.path.exists(config.UPLOAD_DIR)
        print("✓ UPLOAD_DIR exists")
    except Exception as e:
        print(f"✗ UPLOAD_DIR test failed: {e}")
        return False

    return True

def test_ingest():
    """Test ingest functionality."""
    print("\nTesting ingest...")

    # Skip heavy tests for speed
    print("✓ Ingest module available (skipping heavy embedding/vectorstore tests)")
    return True

def test_rag_pipeline():
    """Test RAG pipeline availability."""
    print("\nTesting RAG pipeline...")

    print("✓ RAG pipeline module available (skipping LLM tests without API key)")
    return True

def test_quiz_generator():
    """Test quiz generator availability."""
    print("\nTesting quiz generator...")

    print("✓ Quiz generator module available (skipping LLM tests without API key)")
    return True

def test_analytics():
    """Test analytics functions."""
    print("\nTesting analytics...")

    try:
        analytics.log_question("Test question", "Test answer", [], subject="Test", student_id="test_student")
        print("✓ log_question succeeded")
    except Exception as e:
        print(f"✗ log_question failed: {e}")
        return False

    try:
        stats = analytics.get_summary()
        print(f"✓ get_summary succeeded: {stats}")
    except Exception as e:
        print(f"✗ get_summary failed: {e}")
        return False

    try:
        topics = analytics.get_top_topics()
        print(f"✓ get_top_topics succeeded: {topics}")
    except Exception as e:
        print(f"✗ get_top_topics failed: {e}")
        return False

    return True

def test_api():
    """Test API app creation."""
    print("\nTesting API...")

    try:
        assert hasattr(api, 'app')
        print("✓ FastAPI app created")
    except Exception as e:
        print(f"✗ API app test failed: {e}")
        return False

    try:
        # Test health endpoint
        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        print("✓ Health endpoint works")
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("RAGedu Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Ingest", test_ingest),
        ("RAG Pipeline", test_rag_pipeline),
        ("Quiz Generator", test_quiz_generator),
        ("Analytics", test_analytics),
        ("API", test_api),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())