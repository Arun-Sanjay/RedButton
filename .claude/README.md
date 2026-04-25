# .claude/

Orchestration infrastructure for this Claude Code project.

## Subfolders

**agents/** — Subagent definitions. Each agent is a markdown file with YAML frontmatter defining role, tools, and system prompt. Empty until the project needs specialized agents.

**skills/** — Reusable skill files (instructions Claude Code follows for recurring task types). Each skill is a markdown file. Empty until a task pattern repeats enough to justify codifying.

**commands/** — Slash commands (e.g. /validate, /deploy) invoked during Claude Code sessions. Each is a markdown file defining the command's prompt expansion. Empty until needed.

**prompts/** — Archive of prompts the high-level prompter has produced for this project. Useful for reference and debugging.

**notes/** — Living documentation Claude Code can read for context.
- `decisions.md`: significant architectural or scope decisions.
- `deviations.md`: places implementation diverged from the spec.
