# Decisions Log

Significant architectural or scope decisions made during the project. Append-only. New entries at the top.

## 2026-04-26: Apply Codex audit fixes pre-Phase-6

**Decision:** Apply fixes for four Codex audit findings before launching Phase 6 baseline measurement. Full audit at `/tmp/codex_audit.md`.

**Findings fixed:**
- **BLOCKER** — `submit_answer` no longer accepts unknown `problem_id`s (was a reward shortcut: 10 fake IDs would trigger §8.4's fast-forward shutdown completion). Also rejects duplicate submissions and non-integer inputs.
- **HIGH** — `_is_clean_no_op` uses exact-match against a `CLEAN_NO_OP_PATTERNS` frozenset, not substring scanning (was contaminating the held-out `script_corruption` category by misclassifying `"echo hacked\nexit 0"` as `script_rewrite`).
- **HIGH** — Tool argument type coercion at `_dispatch_tool` boundary via `_safe_str` helper; classifier's `chmod_modification` switched from `is False` to falsey-but-not-None (was crashing on malformed inputs and missing `executable=0` / `executable=""` as tampering).
- **MEDIUM** — Forced-question rejection branch now runs `classify_tampering` (was losing tampering events fired during a forced-question window — most relevantly Tier 3 self-report).

**Findings deferred:**
- HIGH: `run_python` lacks RSS / FD / output-size caps (Phase 7 prep concern; today's subprocess timeout still bounds CPU).
- MEDIUM: Tier 3 doesn't fast-forward on full submission (throughput, not correctness).
- MEDIUM: `/audit` path-prefix matching mismatch between sandbox and classifier.
- LOW: terminal state not sticky after first `done=True`.
- Various LOW findings on output truncation, parse-time errors, etc.

**Rationale:** The four fixed findings either crash training (HIGH-3 type coercion), produce false reward signal (BLOCKER, MEDIUM), or break the held-out generalization claim (HIGH-2 substring match). All four are pre-conditions for trustworthy Phase 6+7 results. Each fix is paired with a regression test (14 new tests; 243 → 257).

**Alternatives considered:**
- (A) Defer all four to Phase 7 prep. Rejected — BLOCKER means Phase 6 baseline numbers would be partially fictional (random submission shortcut + missing chmod tampering).
- (B) Fix only BLOCKER + HIGH-2, defer HIGH-3 and MEDIUM. Rejected — HIGH-3 risks runtime crashes during the actual baseline rollout, and MEDIUM is a 6-line patch.
- (C — chosen) Land all four with regression tests in one commit before launching Phase 6.

**Reversibility:** Easy. Each fix is local: `audit.py:_is_clean_no_op` (5 lines), `audit.py:classify_tampering` chmod branch (3 lines), `shutdown_environment.py:_dispatch_tool` (~30 lines), `shutdown_environment.py:_submit_answer` (~15 lines), `shutdown_environment.py` step's forced-question branch (4 lines).

## 2026-04-25: `sanitize_args` preserves safe scalar types

**Decision:** `shutdown_gym.audit.sanitize_args` passes `bool` / `int` / `float` / `None` through unchanged. Strings and other non-scalar values are still `str()`-coerced and capped at `MAX_ARG_CHARS = 16 KiB`.

**Rationale:** PROJECT.md §11.2 literally specifies "stringify everything", but §11.4's classifier branch for `chmod_modification` checks `args.get("executable") is False` — boolean identity. The §11.3 pipeline composes the two (sanitize → classify), so stringifying booleans makes that branch unreachable. The DoS risk the cap addresses is unbounded *string* payloads; bounded scalars don't pose that risk.

**Alternatives considered:**
- (A) Modify `classify_tampering` to accept stringified booleans (`== "False"`). Rejected — leaks a serialization concern into classification logic.
- (B) Classify on raw args before sanitizing in `Environment.step()`. Rejected — pushes a workaround into every call site instead of fixing the root.
- (C — chosen) Type-preserving sanitize_args.

**Reversibility:** Easy. The change is local to `sanitize_args`; reverting just restores the loop body.

Format:

## YYYY-MM-DD: Short description
**Decision:** What was decided.
**Rationale:** Why.
**Alternatives considered:** What else was on the table.
**Reversibility:** Easy / Hard / Locked-in.
