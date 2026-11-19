# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

**Getting started**

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/microlux.git
cd microlux
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to lint and format the code.

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
pip install -r test/requirements.txt
pytest
```

You can also run only the fast tests (used in CI):

```bash
pytest -m fast
```

Or run tests in parallel:

```bash
pytest -m fast -n 4
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.

---

**Preparing a pull request**

Before creating or updating a pull request, review a few recent merged PRs to
match their title, language, and description style.

- Usually organize the description into `Summary`, `Changes`, and `Tests`; small
  changes may use a shorter description.
- Describe the resulting behavior, any breaking changes and required migration,
  and relevant validation results. Omit development history and redundant details.

---

**What happens next?**

When you open a pull request, GitHub Actions will automatically:
1. Run pre-commit checks (ruff linting and formatting)
2. Run tests with Python 3.10 and 3.12
3. Build documentation (if merging to master)

All checks must pass before your PR can be merged.
