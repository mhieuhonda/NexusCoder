"""
Nexus Agent - AI Agent wrapper
================================
Lớp Agent bao bọc Nexus Coder, cung cấp:
- Quản lý hội thoại
- Personas
- Tool routing (placeholder)
- Logging
"""
from typing import Optional, List, Dict
import json
import os
from datetime import datetime

from ..config import NexusConfig
from ..inference.generator import NexusGenerator, DEFAULT_SYSTEM_PROMPT
from ..tokenizer.tokenizer import NexusTokenizer
from ..model.nexus_coder import NexusCoderForCausalLM
from .. import AUTHOR_INFO


class NexusAgent:
    """AI Agent - wrapper cấp cao cho Nexus Coder."""

    def __init__(
        self,
        generator: Optional[NexusGenerator] = None,
        config: Optional[NexusConfig] = None,
        name: str = "Nexus",
        personality: str = "humorous",
        language: str = "bilingual",
        enable_logging: bool = True,
        log_dir: str = "./logs",
    ):
        self.config = config or NexusConfig()
        self.name = name
        self.personality = personality
        self.language = language
        self.author_info = AUTHOR_INFO

        # Generator
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

        # Tools (placeholder for future tool-use)
        self.tools: Dict[str, callable] = {}

        print(f"✓ Nexus Agent initialized")
        print(f"  Tên: {self.name}")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  Phiên bản: {self.author_info['version']}")

    def register_tool(self, name: str, func: callable) -> None:
        """Đăng ký một tool cho Agent."""
        self.tools[name] = func
        print(f"  🔧 Đăng ký tool: {name}")

    def respond(self, user_input: str, **kwargs) -> str:
        """Phản hồi tin nhắn từ người dùng."""
        start_time = datetime.now()

        try:
            response = self.generator.chat(user_input, **kwargs)
        except Exception as e:
            response = f"⚠️ Xin lỗi, có lỗi xảy ra: {e}"

        elapsed = (datetime.now() - start_time).total_seconds()

        if self.enable_logging:
            self._log_interaction(user_input, response, elapsed)

        return response

    def chat(self) -> None:
        """Bắt đầu chế độ chat tương tác."""
        print("\n" + "=" * 60)
        print(f"  🤖 {self.name} Agent")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  Phiên bản: {self.author_info['version']}")
        print(f"  Ngôn ngữ: {'Song ngữ' if self.language == 'bilingual' else self.language}")
        print("=" * 60)
        print("Gõ 'exit' hoặc 'quit' để thoát.")
        print("Gõ 'reset' để xóa lịch sử trò chuyện.")
        print("Gõ 'info' để xem thông tin model.")
        print("-" * 60 + "\n")

        while True:
            try:
                user_input = input("\n🧑 Bạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Tạm biệt!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print(f"\n👋 Tạm biệt! Hẹn gặp lại bạn. - {self.name}")
                break
            elif user_input.lower() == "reset":
                self.generator.reset_conversation()
                print("\n🔄 Đã xóa lịch sử trò chuyện.")
                continue
            elif user_input.lower() == "info":
                self._print_info()
                continue

            response = self.respond(user_input)
            print(f"\n🤖 {self.name}: {response}")

    def _print_info(self) -> None:
        """In thông tin về model."""
        stats = self.config.estimated_total_params()
        print("\n" + "=" * 60)
        print(f"  Tên: {self.author_info['name']}")
        print(f"  Agent: {self.author_info['agent_name']}")
        print(f"  Tác giả: {self.author_info['name']}")
        print(f"  GitHub: {self.author_info['github']}")
        print(f"  Năm: {self.author_info['year']}")
        print(f"  Phiên bản: {self.author_info['version']}")
        print("-" * 60)
        print(f"  Tổng tham số: {stats['total_params_billion']:.2f}B")
        print(f"  Tham số active: {stats['active_params_billion']:.2f}B")
        print(f"  Context window: {self.config.max_position_embeddings:,} tokens")
        print(f"  Experts: {self.config.num_experts} (active: {self.config.num_active_experts})")
        print(f"  Python: 3.12.13")
        print("=" * 60)

    def _log_interaction(self, user_input: str, response: str, elapsed: float) -> None:
        """Log tương tác vào file."""
        log_file = os.path.join(self.log_dir, f"chat_{datetime.now().strftime('%Y%m%d')}.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": response,
            "elapsed_seconds": elapsed,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_author_info(self) -> Dict[str, str]:
        """Trả về thông tin tác giả."""
        return self.author_info
