# Coding Conventions

**Analysis Date:** 2026-04-08

## Naming Patterns

**Files:**
- Python files use snake_case naming convention (e.g., `document_processor.py`, `vector_store.py`)
- Test files follow `test_*.py` pattern (e.g., `test_document_processor.py`)
- Configuration files use snake_case or kebab-case (e.g., `pyproject.toml`, `.gitignore`)

**Functions:**
- Functions use snake_case (e.g., `process_documents`, `create_vector_store`)
- Private functions prefixed with single underscore (e.g., `_helper_function`)
- Test functions use descriptive snake_case names starting with `test_` (e.g., `test_process_single_document`)

**Variables:**
- Variables use snake_case (e.g., `text_splitter`, `vector_store`)
- Constants use UPPER_SNAKE_CASE (e.g., `CHUNK_SIZE`, `DEFAULT_MODEL`)
- Class attributes use snake_case

**Classes:**
- Classes use PascalCase (e.g., `DocumentProcessor`, `VectorStoreManager`)
- Exception classes follow same pattern with `Error` suffix (e.g., `ProcessingError`)

**Modules/Packages:**
- Package and module names use snake_case (e.g., `rag1`, `app`)
- `__init__.py` files present to mark packages

## Code Style

**Formatting:**
- Code formatted with Black (line length 88) as configured in pyproject.toml
- Import sorting with isort
- Ruff used for linting with configurable line length

**Linting:**
- Ruff configured in pyproject.toml with selective rule enforcement
- Specific rules ignored: E501 (line length), F401 (unused imports), etc.
- Target Python version: 3.9

**Type Hints:**
- Extensive use of type hints throughout codebase
- Function parameters and return types annotated
- Complex types imported from typing module (List, Dict, Optional, etc.)
- Forward references used when necessary (quoted type names)

## Import Organization

**Order:**
1. Standard library imports (e.g., `import os`, `from typing import List`)
2. Third-party imports (e.g., `import streamlit as st`, `from langchain.text_splitter import ...`)
3. Local application imports (e.g., `from .document_processor import ...`)

**Path Aliases:**
- Relative imports used for intra-package references (e.g., `from .utils import ...`)
- No absolute path aliases configured

## Error Handling

**Patterns:**
- Functions raise specific exceptions rather than generic Exception
- Custom exceptions defined in modules where they're used
- Try/except blocks used for specific error conditions
- Errors logged appropriately before re-raising or handling
- Validation functions return boolean results or raise exceptions on invalid input

**Examples:**
- File operations wrapped in try/except for IOError
- External API calls handle connection and timeout errors
- Data validation raises ValueError for invalid inputs

## Logging

**Framework:** 
- Uses Python's built-in logging module (imported as `logger`)
- Streamlit applications use st.success(), st.error(), st.info() for UI feedback
- No external logging framework detected

**Patterns:**
- Logger instances created per module using `logging.getLogger(__name__)`
- Log levels used appropriately (INFO for general flow, ERROR for exceptions, DEBUG for detailed info)
- Log messages include contextual information
- Streamlit apps show user-friendly messages alongside logging

## Comments

**When to Comment:**
- Comments explain complex logic or non-obvious implementation details
- Docstrings used for all public functions, classes, and modules
- Inline comments used sparingly for clarification
- TODO comments used for future work items

**JSDoc/TSDoc:**
- Not applicable (Python codebase)
- Python docstrings follow standard conventions:
  - Triple quotes (""") 
  - Brief summary on first line
  - Detailed description following
  - Args, Returns, Raises sections documented when applicable

## Function Design

**Size:** 
- Functions generally focused and concise (typically < 30 lines)
- Longer functions broken into smaller helper functions when appropriate
- Single responsibility principle observed

**Parameters:** 
- Functions use descriptive parameter names
- Default values provided for optional parameters
- Keyword arguments used for clarity in function calls with multiple parameters
- Type hints provided for all parameters

**Return Values:** 
- Functions return meaningful values or None
- Consistent return types within functions
- Generator functions used for lazy evaluation where appropriate (document processing)
- Tuples used for multiple related return values

## Module Design

**Exports:** 
- Modules define `__all__` explicitly when intended for public API
- Most modules follow implicit export (all non-private names)
- Private functions/variables prefixed with underscore

**Barrel Files:** 
- `__init__.py` files import key components for easier access
- Some packages expose functionality through `__init__.py` (e.g., `from .processor import DocumentProcessor`)
- Not heavily used for re-exporting large numbers of items

## Documentation

**Docstrings:**
- All public modules, classes, and functions have docstrings
- Docstrings follow Google/NumPy style convention with Args, Returns sections
- Module-level docstrings explain purpose and usage
- Class docstrings describe responsibilities and usage patterns

**README:**
- Comprehensive README.md with project overview, setup instructions, usage examples
- Badges for build status, license, etc.
- Clear sections for installation, configuration, and contribution guidelines

**Configuration:**
- Configuration documented in code comments and docstrings
- Environment variables documented where used
- Configuration files (like pyproject.toml) include explanatory comments