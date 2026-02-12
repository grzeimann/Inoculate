# Development style

We use Google-style docstrings across the codebase. New modules and functions
should follow this format for Args/Returns/Raises/Examples blocks.

Example:

```python
def example(a: int, b: float) -> float:
    """Compute a simple result.

    Args:
        a: An integer parameter.
        b: A floating-point parameter.

    Returns:
        The computed floating-point result.

    Raises:
        ValueError: If inputs are invalid.
    """
    if a < 0:
        raise ValueError("a must be non-negative")
    return a + b
```

Notes:
- Keep module-level docstrings brief.
- Prefer type hints for all public functions/classes.
- Avoid executing heavy imports at module import time; use lazy imports if needed.
- Use project exceptions from `inoculate.utils.exceptions` (SchemaError, CheckpointError, ModelSpecError, FitFailureError).
- Follow logging rules from the Production Developer Guide: use module loggers, INFO for stage transitions, DEBUG for internals.
