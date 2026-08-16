"""DateTime Tool - time/date operations."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class DateTimeTool(Tool):
    """DateTime operations: now, parse, format, convert timezone, arithmetic."""
    category = ToolCategory.SYSTEM
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return (
            "DateTime operations: now, parse, format, timezone convert, "
            "date arithmetic, weekday, days between."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "parse", "format", "convert_tz", "add", "diff"],
                    "default": "now",
                },
                "datetime_str": {"type": "string", "description": "For parse/format/convert: input datetime"},
                "format_str": {"type": "string", "description": "strftime/strptime format"},
                "from_tz": {"type": "string", "description": "Source timezone (IANA name)"},
                "to_tz": {"type": "string", "description": "Target timezone"},
                "delta_days": {"type": "integer", "description": "Days to add (can be negative)"},
                "delta_hours": {"type": "integer"},
                "start": {"type": "string", "description": "For diff: start datetime"},
                "end": {"type": "string", "description": "For diff: end datetime"},
            },
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        action = args.get("action", "now")
        
        try:
            if action == "now":
                now_utc = datetime.now(timezone.utc)
                now_local = datetime.now().astimezone()
                return ToolResult(
                    success=True,
                    output=(
                        f"UTC: {now_utc.isoformat()}\n"
                        f"Local: {now_local.isoformat()}\n"
                        f"Timestamp: {now_utc.timestamp()}"
                    ),
                    metadata={
                        "utc": now_utc.isoformat(),
                        "local": now_local.isoformat(),
                        "timestamp": now_utc.timestamp(),
                        "timezone": str(now_local.tzinfo),
                    },
                )
            
            elif action == "parse":
                dt_str = args["datetime_str"]
                fmt = args.get("format_str")
                if fmt:
                    dt = datetime.strptime(dt_str, fmt)
                else:
                    # Try ISO format
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                return ToolResult(
                    success=True,
                    output=f"Parsed: {dt.isoformat()}",
                    metadata={"parsed": dt.isoformat(), "weekday": dt.strftime("%A")},
                )
            
            elif action == "format":
                dt_str = args["datetime_str"]
                fmt = args.get("format_str", "%Y-%m-%d %H:%M:%S")
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                return ToolResult(
                    success=True,
                    output=dt.strftime(fmt),
                    metadata={"format": fmt},
                )
            
            elif action == "convert_tz":
                dt_str = args["datetime_str"]
                from_tz = args.get("from_tz", "UTC")
                to_tz = args["to_tz"]
                
                try:
                    from zoneinfo import ZoneInfo
                    tz_from = ZoneInfo(from_tz)
                    tz_to = ZoneInfo(to_tz)
                except ImportError:
                    return ToolResult(
                        success=False,
                        error="zoneinfo not available (Python 3.9+)",
                        return_code=1,
                    )
                
                dt = datetime.fromisoformat(dt_str).replace(tzinfo=tz_from)
                converted = dt.astimezone(tz_to)
                return ToolResult(
                    success=True,
                    output=f"{dt_str} ({from_tz}) → {converted.isoformat()} ({to_tz})",
                    metadata={"original": dt.isoformat(), "converted": converted.isoformat()},
                )
            
            elif action == "add":
                dt_str = args["datetime_str"]
                delta_days = args.get("delta_days", 0)
                delta_hours = args.get("delta_hours", 0)
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                new_dt = dt + timedelta(days=delta_days, hours=delta_hours)
                return ToolResult(
                    success=True,
                    output=f"{dt.isoformat()} + {delta_days}d {delta_hours}h = {new_dt.isoformat()}",
                    metadata={"result": new_dt.isoformat()},
                )
            
            elif action == "diff":
                start = datetime.fromisoformat(args["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(args["end"].replace("Z", "+00:00"))
                delta = end - start
                return ToolResult(
                    success=True,
                    output=(
                        f"Diff: {delta}\n"
                        f"Days: {delta.days}\n"
                        f"Seconds: {delta.total_seconds()}\n"
                        f"Hours: {delta.total_seconds() / 3600}"
                    ),
                    metadata={
                        "days": delta.days,
                        "seconds": delta.total_seconds(),
                    },
                )
            
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}", return_code=2)
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
