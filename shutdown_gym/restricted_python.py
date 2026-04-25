"""AST-whitelisted Python execution in a spawn-based subprocess.

Two layers of defence:

1. ``validate_ast`` allows only a small set of node types and direct
   builtin calls. No imports, no attribute access, no lambda /
   def / class / try / list-comp.
2. The whitelisted code runs inside a ``multiprocessing`` ``"spawn"``
   subprocess with a hard ``EXEC_TIMEOUT_SECONDS`` deadline. The
   parent kills the child if it exceeds the budget, which prevents
   any infinite loop from pinning a FastAPI worker thread.

``signal.alarm`` is deliberately not used: it only fires on the main
thread, but FastAPI handlers run on worker threads.
"""

import ast
import multiprocessing as mp

ALLOWED_NODES = {
    ast.Expression, ast.Module, ast.Expr,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Name, ast.Load, ast.Store,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Mod, ast.Pow, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Assign,
    ast.If, ast.For, ast.While,
    ast.Call,
}

ALLOWED_BUILTINS = {
    "abs", "min", "max", "sum", "len", "range", "int", "float",
    "str", "round", "sorted", "enumerate", "zip", "all", "any",
    "bool", "list", "dict", "tuple", "set",
}

MAX_CODE_CHARS = 4096
# Bumped 2→3 in Phase 5: macOS spawn-context Process startup occasionally
# spikes to 200–800 ms when the suite runs many subprocess tests back-to-back,
# causing trivial "_result = 1+2" code to time out under a 2-second budget.
# 3 s preserves the functional contract (busy loops still terminate) and
# eliminates the recurring flake observed during Phases 3 and 4.
EXEC_TIMEOUT_SECONDS = 3
RESULT_TRUNCATE = 4096
ERROR_MESSAGE_TRUNCATE = 512


class RestrictedPythonError(Exception):
    """Raised when ``validate_ast`` rejects a node or call."""


def validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise RestrictedPythonError(f"Disallowed node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_BUILTINS:
                    raise RestrictedPythonError(f"Disallowed call: {node.func.id}")
            else:
                raise RestrictedPythonError("Only direct builtin calls allowed")


def _worker_target(code: str, queue) -> None:  # pragma: no cover - runs in subprocess
    """Run inside the spawn-based subprocess. Pushes result or error to queue."""
    try:
        tree = ast.parse(code, mode="exec")
        validate_ast(tree)
    except (SyntaxError, RestrictedPythonError) as e:
        queue.put(f"Error: {e}")
        return

    builtins_dict = (
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    )
    safe_builtins = {b: builtins_dict[b] for b in ALLOWED_BUILTINS}
    safe_globals = {"__builtins__": safe_builtins}
    safe_locals: dict = {}
    try:
        exec(compile(tree, "<sandbox>", "exec"), safe_globals, safe_locals)
        result = safe_locals.get("_result", "OK")
        queue.put(str(result)[:RESULT_TRUNCATE])
    except Exception as e:
        msg = str(e)[:ERROR_MESSAGE_TRUNCATE]
        queue.put(f"Error: {type(e).__name__}: {msg}")


def exec_restricted(code: str) -> str:
    """Validate, run, and return the result string. Never raises."""
    if len(code) > MAX_CODE_CHARS:
        return f"Error: code exceeds {MAX_CODE_CHARS} characters"

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_target, args=(code, queue), daemon=True)
    proc.start()
    proc.join(timeout=EXEC_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return "Error: execution timed out"

    if not queue.empty():
        return queue.get_nowait()
    return "Error: no output"
