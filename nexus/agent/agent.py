"""
Nexus Agent v0.2 - AI Agent với Skills + Tools
===============================================
Major upgrade từ v0.1:
- Tích hợp SkillRegistry (15+ skills)
- Tích hợp ToolRegistry (15+ tools)
- Memory system
- Planner cho multi-step tasks
- Tool routing thông minh
- Safety guardrails
- Audit logging
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any, Callable
import json
import os
from datetime import datetime
from pathlib import Path

from ..config import NexusConfig
from ..inference.generator import NexusGenerator, DEFAULT_SYSTEM_PROMPT
from ..tokenizer.tokenizer import NexusTokenizer
from ..model.nexus_coder import NexusCoderForCausalLM
from .. import AUTHOR_INFO
from ..skills import SkillRegistry, get_global_registry as get_skill_registry
from ..tools import ToolRegistry, ToolContext, get_global_registry as get_tool_registry
from ..safety import SafetyFilter, get_default_guardrails
from .memory import ConversationMemory
from .planner import TaskPlanner
from .router import ToolRouter


class NexusAgent:
    """AI Agent v0.2 - Wrapper cấp cao cho Nexus Coder.
    
    Features:
    - Skill-based routing (15+ skills)
    - Tool use (15+ tools)
    - Conversation memory
    - Task planning
    - Safety guardrails
    - Audit logging
    
    Usage:
        agent = NexusAgent()
        agent.chat()  # Interactive
        # or
        response = agent.respond("Viết hàm fibonacci")
    """
    
    def __init__(
        self,
        generator: Optional[NexusGenerator] = None,
        config: Optional[NexusConfig] = None,
        name: str = "Nexus",
        personality: str = "humorous",
        language: str = "bilingual",
        enable_logging: bool = True,
        log_dir: str = "./logs",
        enable_skills: bool = True,
        enable_tools: bool = True,
        enable_memory: bool = True,
        enable_planner: bool = True,
        enable_safety: bool = True,
        working_dir: str = ".",
    ):
        self.config = config or NexusConfig()
        self.name = name
        self.personality = personality
        self.language = language
        self.author_info = AUTHOR_INFO
        self.working_dir = working_dir
        
        # Generator (model + tokenizer)
        if generator is None:
            self.generator = NexusGenerator(
                model=NexusCoderForCausalLM(self.config),
                tokenizer=NexusTokenizer(),
                config=self.config,
            )
        else:
            self.generator = generator
        
        # Logging
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        if enable_logging:
            os.makedirs(log_dir, exist_ok=True)
        
        # Skills (v0.2 NEW)
        self.enable_skills = enable_skills and self.config.enable_skills
        self.skill_registry: Optional[SkillRegistry] = (
            get_skill_registry() if self.enable_skills else None
        )
        
        # Tools (v0.2 NEW)
        self.enable_tools = enable_tools and self.config.enable_tools
        self.tool_registry: Optional[ToolRegistry] = (
            get_tool_registry() if self.enable_tools else None
        )
        self.tool_router = ToolRouter(self.tool_registry) if self.tool_registry else None
        
        # Memory (v0.2 NEW)
        self.enable_memory = enable_memory and self.config.enable_memory
        self.memory = ConversationMemory() if self.enable_memory else None
        
        # Planner (v0.2 NEW)
        self.enable_planner = enable_planner and self.config.enable_planner
        self.planner = TaskPlanner() if self.enable_planner else None
        
        # Safety (v0.2 NEW)
        self.enable_safety = enable_safety and self.config.enable_safety_filter
        self.safety_filter = SafetyFilter() if self.enable_safety else None
        self.guardrails = get_default_guardrails() if self.enable_safety else None
        
        # Stats
        self._stats = {
            "total_messages": 0,
            "skills_used": 0,
            "tools_called": 0,
            "safety_blocks": 0,
            "session_start": datetime.now().isoformat(),
        }
        
        print(f"✓ Nexus Agent v0.2 initialized")
        print(f"  Tên: {self.name}")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  Phiên bản: {self.author_info['version']}")
        print(f"  Skills: {len(self.skill_registry) if self.skill_registry else 0}")
        print(f"  Tools: {len(self.tool_registry) if self.tool_registry else 0}")
        print(f"  Memory: {'✓' if self.memory else '✗'}")
        print(f"  Planner: {'✓' if self.planner else '✗'}")
        print(f"  Safety: {'✓' if self.safety_filter else '✗'}")
    
    def respond(self, user_input: str, **kwargs) -> str:
        """Phản hồi tin nhắn từ người dùng."""
        start_time = datetime.now()
        self._stats["total_messages"] += 1
        
        # Safety check (input)
        if self.guardrails:
            guard_result = self.guardrails.check(user_input)
            if not guard_result["allowed"]:
                self._stats["safety_blocks"] += 1
                return f"⚠️ {guard_result['message']}"
        
        # Add to memory
        if self.memory:
            self.memory.add(role="user", content=user_input)
        
        # Try skill routing
        skill_used = None
        skill_result = None
        if self.skill_registry:
            from ..skills.base import SkillContext
            ctx = SkillContext(
                prompt=user_input,
                history=self.memory.get_history() if self.memory else [],
                **kwargs,
            )
            skill = self.skill_registry.route(user_input, ctx)
            if skill:
                skill_used = skill.name
                skill_result = skill.execute(ctx)
                self._stats["skills_used"] += 1
        
        # Check for tool calls in user input
        tool_calls_made = []
        if self.tool_router:
            tool_calls = self.tool_router.detect_tool_calls(user_input)
            for tc in tool_calls[:self.config.max_tool_calls]:
                result = self.tool_registry.execute(
                    tc["name"],
                    tc.get("args", {}),
                    ToolContext(working_dir=self.working_dir),
                )
                tool_calls_made.append({
                    "tool": tc["name"],
                    "success": result.success,
                    "output": result.output[:500] if result.output else "",
                })
                self._stats["tools_called"] += 1
        
        # Generate response
        try:
            # Build enhanced prompt with skill/tool context
            enhanced_input = user_input
            if skill_result:
                enhanced_input += f"\n\n[Skill: {skill_used}] {skill_result.output}"
            if tool_calls_made:
                enhanced_input += "\n\n[Tool results:]"
                for tc in tool_calls_made:
                    enhanced_input += f"\n- {tc['tool']}: {tc['output'][:200]}"
            
            response = self.generator.chat(enhanced_input, **kwargs)
        except Exception as e:
            response = f"⚠️ Xin lỗi, có lỗi xảy ra: {e}"
        
        # Add to memory
        if self.memory:
            self.memory.add(role="assistant", content=response)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Logging
        if self.enable_logging:
            self._log_interaction(
                user_input=user_input,
                response=response,
                elapsed=elapsed,
                skill_used=skill_used,
                tools_used=[t["tool"] for t in tool_calls_made],
            )
        
        return response
    
    def chat(self) -> None:
        """Bắt đầu chế độ chat tương tác."""
        print("\n" + "=" * 70)
        print(f"  🤖 {self.name} Agent v0.2.0")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  Phiên bản: {self.author_info['version']}")
        print(f"  Ngôn ngữ: {'Song ngữ' if self.language == 'bilingual' else self.language}")
        print(f"  Skills: {len(self.skill_registry) if self.skill_registry else 0}")
        print(f"  Tools: {len(self.tool_registry) if self.tool_registry else 0}")
        print("=" * 70)
        print("Commands:")
        print("  exit/quit - Thoát")
        print("  reset - Xóa lịch sử")
        print("  info - Thông tin model")
        print("  skills - Liệt kê skills")
        print("  tools - Liệt kê tools")
        print("  stats - Thống kê session")
        print("-" * 70 + "\n")
        
        while True:
            try:
                user_input = input("\n🧑 Bạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Tạm biệt!")
                break
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            if cmd in ["exit", "quit"]:
                print(f"\n👋 Tạm biệt! Hẹn gặp lại bạn. - {self.name}")
                break
            elif cmd == "reset":
                if self.memory:
                    self.memory.clear()
                self.generator.reset_conversation()
                print("\n🔄 Đã xóa lịch sử trò chuyện.")
                continue
            elif cmd == "info":
                self._print_info()
                continue
            elif cmd == "skills":
                self._print_skills()
                continue
            elif cmd == "tools":
                self._print_tools()
                continue
            elif cmd == "stats":
                self._print_stats()
                continue
            
            response = self.respond(user_input)
            print(f"\n🤖 {self.name}: {response}")
    
    def _print_info(self) -> None:
        """In thông tin về model."""
        stats = self.config.estimated_total_params()
        print("\n" + "=" * 60)
        print(f"  Model: {self.author_info['model_name']}")
        print(f"  Agent: {self.author_info['agent_name']}")
        print(f"  Version: {self.author_info['version']}")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  GitHub: {self.author_info['github']}")
        print("-" * 60)
        print(f"  Tổng tham số: {stats['total_params_billion']:.2f}B")
        print(f"  Tham số active: {stats['active_params_billion']:.2f}B")
        print(f"  Context window: {self.config.max_position_embeddings:,} tokens")
        print(f"  Experts: {self.config.num_experts} (active: {self.config.num_active_experts})")
        print(f"  Python: 3.12.13")
        print("=" * 60)
    
    def _print_skills(self) -> None:
        """Liệt kê skills."""
        if not self.skill_registry:
            print("\n❌ Skills chưa được enable")
            return
        print("\n" + "=" * 60)
        print("  Available Skills")
        print("=" * 60)
        by_cat = self.skill_registry.list_by_category()
        for cat, skills in sorted(by_cat.items()):
            print(f"\n  [{cat.upper()}]")
            for s in skills:
                skill = self.skill_registry.get(s)
                print(f"    • {s}: {skill.description}")
        print("\n" + "=" * 60)
    
    def _print_tools(self) -> None:
        """Liệt kê tools."""
        if not self.tool_registry:
            print("\n❌ Tools chưa được enable")
            return
        print("\n" + "=" * 60)
        print("  Available Tools")
        print("=" * 60)
        by_cat = self.tool_registry.list_by_category()
        for cat, tools in sorted(by_cat.items()):
            print(f"\n  [{cat.upper()}]")
            for t in tools:
                tool = self.tool_registry.get(t)
                safety_icon = {
                    "safe": "✓", "moderate": "⚠", "dangerous": "⚡", "destructive": "💀"
                }.get(tool.safety.value, "?")
                print(f"    {safety_icon} {t}: {tool.description}")
        print("\n" + "=" * 60)
    
    def _print_stats(self) -> None:
        """In thống kê session."""
        print("\n" + "=" * 60)
        print("  Session Stats")
        print("=" * 60)
        for k, v in self._stats.items():
            print(f"  {k}: {v}")
        print("=" * 60)
    
    def _log_interaction(
        self,
        user_input: str,
        response: str,
        elapsed: float,
        skill_used: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        """Log tương tác vào file."""
        log_file = os.path.join(self.log_dir, f"chat_{datetime.now().strftime('%Y%m%d')}.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": response,
            "elapsed_seconds": elapsed,
            "skill_used": skill_used,
            "tools_used": tools_used or [],
        }
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def get_author_info(self) -> Dict[str, str]:
        """Trả về thông tin tác giả."""
        return self.author_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Trả về stats."""
        return dict(self._stats)
