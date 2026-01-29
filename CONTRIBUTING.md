# Contributing to TransformPlan

Thank you for your interest in contributing to TransformPlan! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/limebit/transformplan.git
   cd transformplan
   ```

2. Install development dependencies:
   ```bash
   make install-dev
   ```

   This will:
   - Create a virtual environment
   - Install all dependencies including dev tools
   - Set up pre-commit hooks

## Code Style

This project uses strict linting and type checking:

- **Ruff** for linting and formatting (line length: 88)
- **Pyright** for type checking (strict mode)

Before submitting code, ensure it passes all checks:

```bash
make lint    # Run linting and type checking
make format  # Auto-fix formatting issues
```

### Style Guidelines

- Follow PEP 8 conventions
- Use Google-style docstrings
- Add type annotations to all functions
- Keep functions focused and single-purpose

## Running Tests

Run the test suite with:

```bash
make test
```

To run a specific test:

```bash
.venv/bin/pytest tests/test_file.py::test_function -v
```

All new features should include tests. Aim for high test coverage on critical paths.

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run checks** before submitting:
   ```bash
   make lint
   make test
   ```

5. **Submit a pull request** with:
   - Clear description of the changes
   - Reference to any related issues
   - Summary of testing performed

## Adding Dependencies

- Runtime dependencies: `uv add <package>`
- Dev dependencies: `uv add --group dev <package>`

## Architecture Notes

TransformPlan uses a mixin-based design:

- `TransformPlanBase` (core.py) - Execution logic and operation registry
- Operation mixins (ops/) - Column, math, row, string, datetime, and map operations
- `TransformPlan` (plan.py) - Combines all mixins

Operations use deferred execution - they're registered via method chaining, then executed together via `process()`, `validate()`, or `process_checked()`.

## Questions?

If you have questions, please open an issue for discussion.
