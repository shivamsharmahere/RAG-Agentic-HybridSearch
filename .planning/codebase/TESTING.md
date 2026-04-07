# Testing Patterns

**Analysis Date:** 2026-04-08

## Test Framework

**Runner:**
- pytest (configured as dev dependency in pyproject.toml)
- Config: No dedicated config file found; uses pytest defaults

**Assertion Library:**
- Built-in pytest assertions (assert statement)

**Run Commands:**
```bash
pytest              # Run all tests (no test files currently exist)
pytest -v           # Verbose output
pytest --cov=app    # Coverage report (requires pytest-cov, not configured)
```

## Test File Organization

**Location:**
- No test files currently present in codebase
- Based on Python conventions, tests would be located in:
  - `tests/` directory at project root
  - Or alongside source files as `test_*.py`

**Naming:**
- Would follow `test_*.py` pattern for test files
- Test functions would follow `test_*` naming convention

**Structure:**
```
[project-root]/
├── tests/                  # Test directory (not currently present)
│   ├── test_document_processor.py
│   ├── test_rag_pipeline.py
│   └── test_agent.py
└── app/                    # Source code
    └── core/
        ├── document_processor.py
        ├── rag_pipeline.py
        └── agent.py
```

## Test Structure

**Suite Organization:**
- Would use pytest's automatic test discovery
- Test classes optional; functions can be module-level
- Setup/teardown using pytest fixtures

**Patterns:**
```python
# Example pattern that would be used:
import pytest
from app.core.document_processor import validate_file

def test_validate_file_valid_pdf():
    # Arrange
    mock_file = Mock()
    mock_file.size = 1024  # 1KB
    mock_file.name = "test.pdf"
    
    # Act
    is_valid, error_msg = validate_file(mock_file)
    
    # Assert
    assert is_valid == True
    assert error_msg == ""

def test_validate_file_too_large():
    # Arrange
    mock_file = Mock()
    mock_file.size = 15 * 1024 * 1024  # 15MB (>10MB limit)
    mock_file.name = "large.pdf"
    
    # Act
    is_valid, error_msg = validate_file(mock_file)
    
    # Assert
    assert is_valid == False
    assert "exceeds" in error_msg
```

**Setup/Teardown Pattern:**
- Would use `@pytest.fixture` for reusable test setup
- Temporary files/directories handled with `tmpdir` fixture
- Mocking external services with `unittest.mock` or `pytest-mock`

## Mocking

**Framework:**
- Would use `unittest.mock` (built-in) or `pytest-mock` plugin
- No current mocking patterns in codebase (no tests)

**Patterns:**
```python
# External service mocking example:
from unittest.mock import Mock, patch

def test_build_indexes_with_mock():
    with patch('app.core.document_processor.QwenEmbeddings') as mock_embeddings:
        # Configure mock
        mock_embeddings.return_value = Mock()
        
        # Test function
        result = build_indexes([Mock()])
        
        # Verify mock was called
        mock_embeddings.assert_called_once()
```

**What to Mock:**
- External API calls (Groq, Tavily, embedding services)
- File system operations
- Streamlit UI components
- Database/vector store connections

**What NOT to Mock:**
- Pure utility functions
- Internal business logic with no external dependencies
- Data transformation functions

## Fixtures and Factories

**Test Data:**
- Would use factories for creating test documents
- Sample PDF content for testing document processing
- Mock Streamlit uploaded file objects

**Location:**
- Test fixtures would be in `tests/conftest.py` or alongside test files
- Sample test data in `tests/fixtures/` or `tests/sample_data/`

**Example Factory Pattern:**
```python
import pytest
from langchain.schema import Document

@pytest.fixture
def sample_document():
    return Document(
        page_content="This is a test document.",
        metadata={"file_name": "test.pdf", "file_size": 1024}
    )

@pytest.fixture
def multiple_documents():
    return [
        Document(page_content="First test document.", metadata={"file_name": "doc1.pdf"}),
        Document(page_content="Second test document.", metadata={"file_name": "doc2.pdf"}),
    ]
```

## Coverage

**Requirements:**
- No coverage requirements currently enforced
- Dev dependencies include pytest but not pytest-cov

**View Coverage:**
- Would use: `pytest --cov=app --cov-report=html`
- No current coverage configuration

## Test Types

**Unit Tests:**
- Would test individual functions in isolation
- Focus on document processing, validation, and utility functions
- Mock external dependencies

**Integration Tests:**
- Would test interactions between components (e.g., document processing → indexing)
- May use temporary directories for file operations
- Test end-to-end flows with mocked external services

**E2E Tests:**
- Not currently implemented
- Would test full Streamlit application workflows
- Likely using Selenium or Playwright for browser automation

## Common Patterns

**Async Testing:**
- Not applicable (codebase is primarily synchronous)
- If async functions were added, would use `pytest.mark.asyncio`

**Error Testing:**
```python
def test_validate_file_invalid_type():
    mock_file = Mock()
    mock_file.size = 1024
    mock_file.name = "test.txt"  # Not PDF
    
    is_valid, error_msg = validate_file(mock_file)
    
    assert is_valid == False
    assert "not a PDF" in error_msg

def test_build_indexes_no_chunks():
    with pytest.raises(ValueError, match="No document chunks provided"):
        build_indexes([])
```

**Streamlit Testing:**
- Would use `streamlit-runner` or similar for testing UI components
- Test session state manipulations
- Verify UI elements are rendered correctly

## Development Practices

**Test-Driven Development:**
- Not currently practiced (no test files exist)
- Would follow Red-Green-Refactor cycle when implemented

**Continuous Integration:**
- No CI configuration found in repository
- Tests would be run on push/PR in CI pipeline

**Pre-commit Hooks:**
- No pre-commit configuration found
- Would potentially run tests on pre-commit when implemented

## Recommendations

1. **Create tests directory:** Establish `tests/` directory at project root
2. **Add core module tests:** Start with document processor and validation functions
3. **Implement fixtures:** Create reusable fixtures for mock files and documents
4. **Add coverage configuration:** Configure pytest-cov for coverage reporting
5. **Establish testing conventions:** Document patterns for mocking, fixtures, and test organization