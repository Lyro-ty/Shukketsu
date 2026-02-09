Run the full verification suite for the current implementation state:

1. **Ruff lint**: `ruff check code/ tests/`
2. **Ruff format**: `ruff format --check code/ tests/`
3. **Mypy**: `mypy code/shukketsu/`
4. **Unit tests**: `pytest tests/unit/ -v`
5. **Integration tests** (only if requested): `pytest tests/integration/ -v -m integration`

For each check, report pass/fail and show specific errors if any.
Summarize at the end: how many passed, what needs fixing.

If the user specified a step number or scope: $ARGUMENTS
