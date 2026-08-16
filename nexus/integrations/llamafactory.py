"""
LlamaFactory-inspired dataset format converters for Nexus Coder v0.3
====================================================================
Ported & simplified from hiyouga/LlamaFactory (Apache 2.0).

Converts between popular supervised-fine-tuning (SFT) data formats so
Nexus Coder can train on data collected from any of them.

Supported formats:
  - alpaca      {instruction, input, output}
  - sharegpt    {conversations: [{from, value}]}
  - chatml      {messages: [{role, content}]}
  - openai      {messages: [{role, content}]}  (same as chatml)
  - completion  {prompt, completion}

All converters return a unified dict: {system, user, assistant}
(matching Nexus Coder's internal training format).

Original attribution:
    LlamaFactory: Unify Fine-tuning 100+ LLMs.
    Author: hiyouga
    License: Apache 2.0
    Source: https://github.com/hiyouga/LlamaFactory
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Iterator


def alpaca_to_nexus(example: Dict) -> Dict[str, str]:
    """{instruction, input, output} → {system, user, assistant}"""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    user = f"{instruction}\n\nInput: {inp}" if inp else instruction
    return {
        "system": example.get("system_prompt", ""),
        "user": user.strip(),
        "assistant": out.strip(),
    }


def sharegpt_to_nexus(example: Dict) -> List[Dict[str, str]]:
    """{conversations: [{from, value}]} → list of {system, user, assistant} turns.
    A single ShareGPT conversation may produce multiple Q/A turns.
    """
    conv = example.get("conversations", [])
    system = example.get("system", "")
    turns: List[Dict[str, str]] = []
    current_user: Optional[str] = None
    for msg in conv:
        role = msg.get("from", "").lower()
        value = msg.get("value", "")
        if role in ("human", "user"):
            if current_user is not None:
                # No assistant reply, push anyway with empty assistant
                turns.append({"system": system, "user": current_user, "assistant": ""})
            current_user = value
        elif role in ("gpt", "assistant", "bot"):
            if current_user is None:
                continue
            turns.append({"system": system, "user": current_user, "assistant": value})
            current_user = None
        elif role == "system":
            system = value
    if current_user is not None:
        turns.append({"system": system, "user": current_user, "assistant": ""})
    return turns


def chatml_to_nexus(example: Dict) -> List[Dict[str, str]]:
    """{messages: [{role, content}]} → list of {system, user, assistant} turns."""
    messages = example.get("messages", [])
    system = ""
    turns: List[Dict[str, str]] = []
    current_user: Optional[str] = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            if current_user is not None:
                turns.append({"system": system, "user": current_user, "assistant": ""})
            current_user = content
        elif role == "assistant":
            if current_user is None:
                continue
            turns.append({"system": system, "user": current_user, "assistant": content})
            current_user = None
    if current_user is not None:
        turns.append({"system": system, "user": current_user, "assistant": ""})
    return turns


def completion_to_nexus(example: Dict) -> Dict[str, str]:
    """{prompt, completion} → {system, user, assistant}"""
    return {
        "system": "",
        "user": example.get("prompt", ""),
        "assistant": example.get("completion", ""),
    }


def detect_format(example: Dict) -> str:
    """Auto-detect the SFT format of an example."""
    if "conversations" in example:
        return "sharegpt"
    if "messages" in example:
        return "chatml"
    if "instruction" in example:
        return "alpaca"
    if "prompt" in example and "completion" in example:
        return "completion"
    raise ValueError(f"Unknown SFT format. Keys: {list(example.keys())}")


def convert_to_nexus(example: Dict) -> List[Dict[str, str]]:
    """Auto-detect format and convert to Nexus unified format.
    Returns a list of turns (most formats produce 1 turn; ShareGPT/ChatML may produce multiple).
    """
    fmt = detect_format(example)
    if fmt == "alpaca":
        return [alpaca_to_nexus(example)]
    if fmt == "sharegpt":
        return sharegpt_to_nexus(example)
    if fmt == "chatml":
        return chatml_to_nexus(example)
    if fmt == "completion":
        return [completion_to_nexus(example)]
    return []


def stream_jsonl(path: str) -> Iterator[Dict[str, str]]:
    """Stream-convert a JSONL file in any SFT format to Nexus examples.
    Yields {system, user, assistant} dicts lazily — safe for large files.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for turn in convert_to_nexus(obj):
                yield turn


__all__ = [
    "alpaca_to_nexus",
    "sharegpt_to_nexus",
    "chatml_to_nexus",
    "completion_to_nexus",
    "detect_format",
    "convert_to_nexus",
    "stream_jsonl",
]
