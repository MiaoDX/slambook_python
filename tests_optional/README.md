# Optional Backend Checks

These tests validate that the optional dependency groups declared in
`pyproject.toml` are usable together. They are intentionally outside the
default `tests/` path so the core educational test suite stays deterministic.

Run them after installing all extras:

```bash
uv sync --all-extras --frozen
uv run --all-extras --frozen python -m pytest tests_optional
```

In mainland China, use a PyPI mirror for package downloads:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --all-extras --frozen
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --all-extras --frozen python -m pytest tests_optional
```

Missing optional packages are reported as pytest skips in this suite; that is
expected when the environment was installed with only `core` and `test`.
