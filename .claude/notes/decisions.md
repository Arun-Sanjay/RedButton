# Decisions Log

Significant architectural or scope decisions made during the project. Append-only. New entries at the top.

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
