# Repository Guidelines

## Purpose

microlux provides differentiable binary-microlensing magnification and
trajectory calculations in JAX.

## Structure and Sources of Truth

- Library code and API docstrings: [src/microlux/](src/microlux/).
- Tests: [test/](test/).
- Usage examples and API documentation: [docs/](docs/).
- Development setup, formatting, testing, and documentation workflows:
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Rules

- Add `@pytest.mark.fast` to new routine regression tests, including accuracy
  and gradient tests, so CI does not silently skip them.
- Keep intentionally slow or manual tests outside the `fast` selection.
- Keep Quick Start focused on runnable examples; put detailed parameter,
  reference-frame, and numerical-boundary explanations in API docstrings.
- Use Google-style syntax for structured docstring sections parsed by
  mkdocstrings (see [mkdocs.yml](mkdocs.yml)). Give each parameter or attribute
  its own `name (type): description` entry, with continuation lines indented.

## Validation

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for checks relevant to the change
and pull request preparation.
CI selects only `fast` tests; see [.github/workflows/run_test.yml](.github/workflows/run_test.yml).
After adding tests, verify collection and run the affected tests:

```bash
pytest --collect-only -q -m fast test/<test_file>.py
pytest -q -m fast test/<test_file>.py
```

For docstring changes, build the docs following the contribution guide and
inspect the rendered API tables; a successful build alone does not verify layout.
