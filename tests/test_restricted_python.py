import ast
import time

import pytest

from shutdown_gym.restricted_python import (
    EXEC_TIMEOUT_SECONDS,
    MAX_CODE_CHARS,
    RestrictedPythonError,
    exec_restricted,
    validate_ast,
)

# ---- AST validation: allowed ----


@pytest.mark.parametrize(
    "code",
    [
        "_result = 1 + 2",
        "_result = (1 < 2)",
        "_result = [1, 2, 3]",
        "_result = {'a': 1}",
        "_result = (1, 2)",
        "_result = {1, 2, 3}",
        "x = 0\nfor i in range(5):\n    x = x + i\n_result = x",
        "i = 0\nwhile i < 3:\n    i = i + 1\n_result = i",
        "_result = str(42)",
        "_result = float(1)",
        "_result = sum(range(5))",
        "_result = len([1, 2, 3])",
        "_result = abs(-7)",
        "_result = max(1, 2, 3)",
        "_result = min(1, 2, 3)",
        "_result = sorted([3, 1, 2])",
        "_result = -5",
        "_result = not False",
    ],
)
def test_validate_ast_allowed(code):
    tree = ast.parse(code, mode="exec")
    validate_ast(tree)


# ---- AST validation: forbidden node types ----


@pytest.mark.parametrize(
    "code,fragment",
    [
        ("import os", "Import"),
        ("from os import path", "ImportFrom"),
        ("a = b.c", "Attribute"),
        ("lambda x: x", "Lambda"),
        ("def f():\n    pass", "FunctionDef"),
        ("class X:\n    pass", "ClassDef"),
        ("[x for x in range(5)]", "ListComp"),
        ("try:\n    pass\nexcept Exception:\n    pass", "Try"),
        ("with open('x') as f:\n    pass", "With"),
        ("yield 1", "Yield"),
    ],
)
def test_validate_ast_disallowed_node(code, fragment):
    tree = ast.parse(code, mode="exec")
    with pytest.raises(RestrictedPythonError) as exc_info:
        validate_ast(tree)
    assert fragment in str(exc_info.value)


# ---- AST validation: forbidden builtin/free-name calls ----


@pytest.mark.parametrize(
    "code",
    [
        "open('/tmp/x')",
        "exec('print(1)')",
        "eval('1+1')",
        "getattr({}, 'keys')",
        "__import__('os')",
        "print('hi')",
    ],
)
def test_validate_ast_disallowed_call(code):
    tree = ast.parse(code, mode="exec")
    with pytest.raises(RestrictedPythonError) as exc_info:
        validate_ast(tree)
    assert "Disallowed call" in str(exc_info.value)


# ---- end-to-end via exec_restricted (subprocess) ----


def test_exec_returns_result_value():
    assert exec_restricted("_result = 1 + 2") == "3"


def test_exec_default_result_when_underscore_result_unset():
    assert exec_restricted("x = 1") == "OK"


def test_exec_oversized_code_rejected_without_subprocess():
    huge = "x = 1\n" * MAX_CODE_CHARS  # ~6× MAX_CODE_CHARS
    out = exec_restricted(huge)
    assert "exceeds" in out
    assert str(MAX_CODE_CHARS) in out


def test_exec_syntax_error_returned_as_error():
    out = exec_restricted("1 +")
    assert out.startswith("Error:")


def test_exec_disallowed_node_returned_as_error():
    out = exec_restricted("import os")
    assert "Disallowed" in out


def test_exec_disallowed_attribute_blocked_end_to_end():
    """Attribute access blocks at exec_restricted level — important
    because it's the easiest path back to the parent process via
    ``__builtins__.__import__`` etc.
    """
    out = exec_restricted("x = 1\n_result = x.real")
    assert "Disallowed" in out


def test_exec_timeout_fires_on_busy_loop():
    """Busy ``while True`` must hit the subprocess timeout, not pin the
    test process.
    """
    start = time.monotonic()
    out = exec_restricted("while True:\n    x = 1")
    elapsed = time.monotonic() - start
    assert "timed out" in out
    # Spawn-context process startup is slow on macOS; allow generous slack.
    assert elapsed < EXEC_TIMEOUT_SECONDS + 5.0
