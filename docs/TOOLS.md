# Tools Documentation

Nexus Coder v0.2 có 18+ tools để tương tác với môi trường.

## Safety Levels

| Level | Icon | Description |
|-------|------|-------------|
| SAFE | ✓ | Read-only, no side effects |
| MODERATE | ⚠ | Writes to local files |
| DANGEROUS | ⚡ | Executes commands, network ops |
| DESTRUCTIVE | 💀 | Can delete data, requires confirmation |

## Tools by Category

### FILE Operations
- `file_read` (✓) - Đọc file text
- `file_write` (⚠) - Ghi file (overwrite/append)
- `file_list` (✓) - Liệt kê files với glob
- `file_delete` (💀) - Xóa file/thư mục

### EXEC
- `shell_exec` (⚡) - Execute bash commands
- `python_exec` (⚡) - Execute Python code (sandboxed)
- `git_ops` (⚡) - Git commands

### WEB
- `http_request` (⚠) - HTTP GET/POST/PUT/DELETE
- `web_fetch` (✓) - Fetch webpage, extract text
- `web_search` (✓) - Search web

### CODE
- `code_search` (✓) - Regex search trong code
- `code_lint` (✓) - Lint code (ruff, flake8, pylint)
- `code_format` (⚠) - Format code (black, autopep8, isort)
- `regex_search` (✓) - Regex search trong files

### MATH
- `calculator` (✓) - Safe math expression eval

### PARSER
- `json_parse` (✓) - Parse JSON với query support
- `yaml_parse` (✓) - Parse YAML
- `csv_parse` (✓) - Parse CSV

### SYSTEM
- `datetime` (✓) - DateTime operations + timezone

### NETWORK
- `dns_lookup` (✓) - DNS lookup (A, AAAA, MX, NS, CNAME, TXT)
- `ping` (✓) - Ping host

### CRYPTO
- `hash` (✓) - Compute hash (md5, sha1, sha256, sha512, blake2)
- `encrypt` (⚡) - AES-256-GCM encrypt/decrypt

### FILE (Archive)
- `archive` (⚠) - ZIP/TAR create/extract/list

## Usage

```python
from nexus.tools import get_global_registry, ToolContext

registry = get_global_registry()

# List all tools
print(registry.list_tools())

# Execute tool
from nexus.tools.base import ToolContext
ctx = ToolContext(working_dir="/tmp")
result = registry.execute("file_read", {"path": "/etc/hostname"}, ctx)
print(result.output)

# Check safety
tool = registry.get("file_delete")
print(f"Safety: {tool.safety.value}")
```

## Audit Log

All tool calls are logged to `./logs/tool_audit.jsonl`:

```json
{
  "timestamp": 1234567890.123,
  "tool": "file_write",
  "safety": "moderate",
  "args": {"path": "/tmp/test.txt", "content": "hello"},
  "working_dir": ".",
  "user_id": null,
  "success": true,
  "return_code": 0,
  "duration": 0.001
}
```

## Safety Features

1. **Confirmation required** for DANGEROUS and DESTRUCTIVE tools
2. **Dry-run mode** to preview actions without executing
3. **Pre-hooks** for rate limiting, auth checks
4. **Post-hooks** for metrics, notifications
5. **Audit log** for compliance
6. **Blocked commands** for known dangerous patterns

## Custom Tools

```python
from nexus.tools.base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety

class MyTool(Tool):
    category = ToolCategory.FILE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "My custom tool"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
    
    def execute(self, args, context):
        return ToolResult(
            success=True,
            output=f"Processed: {args['input']}",
        )

# Register
from nexus.tools import get_global_registry
get_global_registry().register(MyTool())
```

## Tool Calling via Natural Language

Agent có thể detect tool calls từ natural language:

- "read file /etc/hostname" → `file_read`
- "run ls -la" → `shell_exec`
- "search for TODO in src/" → `regex_search`
- "fetch https://example.com" → `web_fetch`
- "git status" → `git_ops`

Hoặc JSON format:
```json
{"tool": "file_read", "args": {"path": "/etc/hostname"}}
```

Or @-mention:
```
@file_read path=/etc/hostname
```
