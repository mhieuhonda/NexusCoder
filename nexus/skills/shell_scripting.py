"""Shell Scripting Skill - Sinh bash/sh script với error handling chuẩn.

Cung cấp template với: set -euo pipefail, trap cleanup, logging functions,
arg parsing (getopts), signal handling, retry logic, và parallel execution.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class ShellScriptingSkill(Skill):
    """Sinh bash/zsh/sh script với best practices."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "bash", "shell script", "sh", "zsh",
        "shell", "script bash", "viết shell script",
        "bash script", "command line script", "make script",
    ]
    examples = [
        "Write a bash script to backup a directory",
        "Bash script with argparse and logging",
        "Shell script to deploy and rollback on failure",
    ]

    @property
    def name(self) -> str:
        return "shell_scripting"

    @property
    def description(self) -> str:
        return (
            "Sinh bash/zsh/sh script với set -euo pipefail, trap cleanup, "
            "logging, arg parsing, retry logic, và parallel execution."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        if any(k in prompt_lower for k in (".sh", "bash", "zsh", "shell")):
            score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[ShellScripting] Production-ready bash template ready.",
            artifacts=[
                {"path": "shell/template.sh", "content": _SHELL_TEMPLATE},
                {"path": "shell/best_practices.md", "content": _BEST_PRACTICES},
            ],
            metadata={
                "skill": self.name,
                "features": {
                    "strict_mode": "set -euo pipefail — fail fast on errors",
                    "cleanup_trap": "trap cleanup EXIT INT TERM — auto cleanup on exit/signal",
                    "logging": "log_info / log_warn / log_error with timestamps + levels",
                    "arg_parsing": "getopts for short flags + manual long-option parsing",
                    "temp_files": "mktemp + trap rm — never leave temp files",
                    "retry": "with_retry N command — exponential backoff",
                    "parallel": "xargs -P / GNU parallel / background jobs + wait",
                    "subshell_isolation": "( cmd1; cmd2 ) > output — contained side effects",
                },
                "portability": {
                    "shebang": "#!/usr/bin/env bash  (preferred over #!/bin/bash)",
                    "sh_compat": "Use #!/bin/sh if POSIX-only features used",
                    "bash_version_guard": "if (( BASH_VERSINFO[0] < 4 )); then exit 1; fi",
                    "avoid_bashisms_in_sh": "no [[ ]], no arrays, no $(< file)",
                },
                "safety": [
                    "Quote ALL variables: \"$var\" not $var",
                    "Use ${var:-default} for defaults",
                    "Use 'read -r' not 'read' (preserve backslashes)",
                    "Avoid eval / $() with untrusted input — code injection",
                    "Use mktemp -d for temp directories, not /tmp/$$",
                    "Pin 'set -o pipefail' to catch pipe failures",
                    "Use 'command -v' to check binaries, not 'which'",
                ],
                "linting": {
                    "shellcheck": "shellcheck script.sh  — catches common bugs",
                    "shfmt": "shfmt -i 4 -ci script.sh  — formatting",
                    "ci": "Run both in pre-commit hook",
                },
            },
            suggestions=[
                "Specify shell: bash (default), zsh, or POSIX sh",
                "Indicate if script is for CI or interactive use",
                "Mention required binaries — script will check via command -v",
                "Ask for retry / parallel features if applicable",
            ],
        )


_SHELL_TEMPLATE = '''#!/usr/bin/env bash
# <one-line description of script>
#
# Usage:  script.sh [-v] [--dry-run] [-o OUTPUT] <input>
#
# Author: Hieu Louis (2026)
# License: MIT

set -euo pipefail
IFS=$'\\n\\t'

# ---- Config -----------------------------------------------------------------
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TMP_DIR="$(mktemp -d)"
readonly LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/${SCRIPT_NAME%.sh}.log}"

# ---- Cleanup ----------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ "${KEEP_TMP:-}" != "1" ]]; then
        rm -rf "$TMP_DIR" 2>/dev/null || true
    fi
    log_info "Exit code: $exit_code"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM HUP

# ---- Logging ----------------------------------------------------------------
_log() {
    local level="$1"; shift
    local ts; ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local msg="$*"
    printf '[%s] [%s] [%s] %s\\n' "$ts" "$level" "$$" "$msg" | tee -a "$LOG_FILE" >&2
}
log_info()  { _log INFO  "$@"; }
log_warn()  { _log WARN  "$@"; }
log_error() { _log ERROR "$@"; }
log_debug() { [[ "${VERBOSE:-}" == "1" ]] && _log DEBUG "$@" || true; }

# ---- Error helpers ---------------------------------------------------------
die() {
    log_error "$*"
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

# ---- Retry -----------------------------------------------------------------
with_retry() {
    local attempts="$1"; shift
    local delay="${RETRY_DELAY:-2}"
    local attempt=1
    while (( attempt <= attempts )); do
        log_debug "Attempt $attempt/$attempts: $*"
        if "$@"; then return 0; fi
        attempt=$((attempt + 1))
        sleep "$delay"
        delay=$((delay * 2))   # exponential backoff
    done
    die "Command failed after $attempts attempts: $*"
}

# ---- Arg parsing -----------------------------------------------------------
VERBOSE=0
DRY_RUN=0
OUTPUT=""
INPUT=""

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [options] <input>

Options:
  -h, --help        Show this help and exit
  -v, --verbose     Verbose logging
      --dry-run     Print actions, do not execute
  -o, --output FILE Output file (default: stdout)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage; exit 0 ;;
        -v|--verbose) VERBOSE=1 ;;
        --dry-run)    DRY_RUN=1 ;;
        -o|--output)  OUTPUT="$2"; shift ;;
        --) shift; break ;;
        -*) die "Unknown option: $1" ;;
        *)  INPUT="$1"; break ;;
    esac
    shift
done

[[ -n "$INPUT" ]] || { usage; die "Missing <input> argument"; }
[[ -f "$INPUT" ]] || die "Input file not found: $INPUT"

# ---- Main ------------------------------------------------------------------
main() {
    log_info "Starting $SCRIPT_NAME"
    log_info "Input: $INPUT | Output: ${OUTPUT:-stdout} | Dry-run: $DRY_RUN"

    require_cmd "curl"
    require_cmd "jq"

    # Example: process input
    local line_count
    line_count=$(wc -l < "$INPUT")
    log_info "Input has $line_count lines"

    if (( DRY_RUN )); then
        log_info "[dry-run] Would process $line_count lines"
        return 0
    fi

    # Example: parallel processing (4 workers)
    xargs -P 4 -I {} sh -c '
        line="$1"
        # process "$line"
        echo "processed: $line"
    ' _ {} < "$INPUT" > "${OUTPUT:-/dev/stdout}"

    log_info "Done."
}

main "$@"
'''


_BEST_PRACTICES = """# Shell Scripting Best Practices

## Strict Mode (always start with this)
```bash
set -euo pipefail
IFS=$'\\n\\t'
```
- `-e`: exit on error
- `-u`: undefined variable is error
- `-o pipefail`: pipe fails if any command fails
- `IFS=$'\\n\\t'`: prevent word-splitting on spaces

## Cleanup Trap
```bash
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT INT TERM
```
- Runs on normal exit, Ctrl-C, kill, SIGHUP.
- Never use `$$` for temp names — predictable path = security risk.

## Logging
- Use stderr for logs (`>&2`) — keep stdout clean for data.
- ISO-8601 UTC timestamps: `date -u +"%Y-%m-%dT%H:%M:%SZ"`.
- Tee to log file for audit trail.

## Variables
- Always quote: `"$VAR"` not `$VAR` — handles spaces in paths.
- Use `${VAR:-default}` for defaults.
- Use `${VAR:?message}` to require a variable.

## Subshells vs Functions
- Subshell `( cmd )` isolates cwd changes / variable mutations.
- Function `f() { ... }` shares state with caller.

## Arg Parsing
- Short flags: `getopts "vo:" opt` (built-in, POSIX).
- Long flags: manual `case` loop (more flexible, bash 4+).

## Signals
- `trap func INT TERM EXIT` — handle Ctrl-C, kill, normal exit.
- `kill -0 $pid` checks if process is alive (without sending signal).

## Parallelism
- `xargs -P N`: simplest parallel executor.
- GNU parallel: more features (progress, resume, retries).
- Background + wait: `cmd1 & cmd2 & wait` — no progress visibility.

## Idempotency
- Scripts should be safe to re-run (CI reruns on flaky failures).
- Use `mkdir -p`, `touch`, `[[ -f ]] || create` guards.

## Linting (mandatory in CI)
- `shellcheck` — catches 200+ common bugs.
- `shfmt -i 4 -ci -bn` — formatting (indent, case indent, binary ops).

## When NOT to use bash
- String parsing with regex -> use Python / Perl.
- JSON manipulation -> use `jq` or Python.
- Date arithmetic -> use Python / `date -d` (carefully).
- Anything > 50 lines of logic -> consider rewriting in Python.
"""
