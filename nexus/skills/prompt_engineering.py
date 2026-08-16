"""Prompt Engineering Skill - Bộ sưu tập prompt templates.

Cung cấp templates cho: Zero-shot, Few-shot, Chain-of-Thought (CoT),
Self-Consistency, ReAct, Role-play, Tree-of-Thoughts, Retrieval-Augmented.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class PromptEngineeringSkill(Skill):
    """Sinh prompt templates cho các technique phổ biến."""

    category = SkillCategory.LANGUAGE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "prompt", "prompt engineering", "few-shot", "chain of thought",
        "cot", "zero-shot", "react", "tree of thought",
        "self-consistency", "role-play", "prompt template",
        "system prompt", "kỹ thuật prompt", "thiết kế prompt",
    ]
    examples = [
        "Design a few-shot prompt for sentiment classification",
        "Write a Chain-of-Thought prompt for math reasoning",
        "Build a ReAct agent prompt with tool use",
    ]

    @property
    def name(self) -> str:
        return "prompt_engineering"

    @property
    def description(self) -> str:
        return (
            "Bộ sưu tập prompt templates: zero-shot, few-shot, CoT, "
            "ReAct, ToT, self-consistency, role-play, RAG."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.16
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[PromptEngineering] Template collection ready (8 techniques).",
            artifacts=[
                {"path": "prompts/templates.md", "content": _PROMPT_TEMPLATES},
                {"path": "prompts/best_practices.md", "content": _BEST_PRACTICES},
            ],
            metadata={
                "skill": self.name,
                "techniques": {
                    "zero_shot": {
                        "use_when": "Task is well-known to the model, simple",
                        "structure": "Instruction + input -> output",
                        "cost": "Lowest token count",
                    },
                    "few_shot": {
                        "use_when": "Output format is strict, edge cases need examples",
                        "structure": "Instruction + K examples (I/O pairs) + new input",
                        "k_typical": "2-5 (diminishing returns past 8)",
                    },
                    "chain_of_thought": {
                        "use_when": "Multi-step reasoning (math, logic, code)",
                        "structure": "Add 'Let's think step by step.' or show reasoning trace",
                        "variants": ["Zero-shot CoT", "Few-shot CoT", "Auto-CoT"],
                    },
                    "self_consistency": {
                        "use_when": "Single CoT can be wrong; want robustness",
                        "structure": "Sample N CoT paths, majority vote on final answer",
                        "cost": "N x CoT cost",
                    },
                    "react": {
                        "use_when": "Agent needs to call external tools",
                        "structure": "Thought -> Action -> Observation loop",
                        "tools_needed": "Search, calculator, code interpreter, etc.",
                    },
                    "tree_of_thoughts": {
                        "use_when": "Search space large, exploration valuable",
                        "structure": "Generate K candidate thoughts, evaluate, branch",
                        "cost": "High — exponential in depth",
                    },
                    "role_play": {
                        "use_when": "Need expert persona for nuanced domain",
                        "structure": "'You are a <expert> with <background>. ...'",
                        "pitfall": "Persona alone doesn't substitute for knowledge",
                    },
                    "rag": {
                        "use_when": "Need current / proprietary knowledge",
                        "structure": "Retrieve K docs -> stuff in prompt -> answer",
                        "retrieval": "BM25, dense embeddings, hybrid",
                    },
                },
                "anti_patterns": [
                    "Vague verbs: 'do something good' -> specify metric",
                    "Conflicting constraints: 'be concise but detailed'",
                    "Over-long context: dilutes signal (>4k tokens often hurts)",
                    "Trusting outputs without verification (no self-check)",
                    "Prompt injection vulnerability (esp. in RAG)",
                ],
                "evaluation": {
                    "metrics": ["exact_match", "f1", "bleu", "rouge", "pass@k (code)"],
                    "methods": ["held-out set", "LLM-as-judge", "human eval", "A/B test"],
                    "tools": ["promptfoo", "langsmith", "helicone", "traceloop"],
                },
            },
            suggestions=[
                "Specify the task and model (GPT-4 / Claude / Llama)",
                "Indicate whether few-shot examples are available",
                "Mention token budget (affects technique choice)",
                "Ask for evaluation plan if going to production",
            ],
        )


_PROMPT_TEMPLATES = '''# Prompt Engineering Templates

## 1. Zero-Shot
```
You are a helpful assistant. {task_instruction}

Input: {input}
Output:
```

## 2. Few-Shot (K=3)
```
Classify the sentiment of each review as positive, negative, or neutral.

Review: "The food was amazing, best meal of my life!"
Sentiment: positive

Review: "Service was slow, but the manager comped our meal."
Sentiment: neutral

Review: "Worst experience ever. Will never come back."
Sentiment: negative

Review: "{input_review}"
Sentiment:
```

## 3. Chain-of-Thought (CoT) — Zero-Shot
```
{task_instruction}

Think step by step before giving your final answer.

Question: {question}

Step-by-step reasoning:
```

## 4. Chain-of-Thought (CoT) — Few-Shot
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls?
A: Roger starts with 5 balls. 2 cans x 3 balls = 6 more. Total = 5 + 6 = 11.
Final answer: 11

Q: {new_question}
A:
```

## 5. Self-Consistency
```
[Run K independent CoT prompts with temperature=0.7]

Final answer = majority vote of K answers.
Recommended K: 5-40 (more = better, expensive).
```

## 6. ReAct (Reasoning + Acting)
```
Answer the following question using the available tools.

Available tools:
- search(query: str) -> str: web search
- calculator(expr: str) -> str: math evaluation

Question: {question}

Thought 1: I need to find ... I'll search for ...
Action 1: search("{query_1}")
Observation 1: {result_1}

Thought 2: Now I need to compute ... I'll use the calculator.
Action 2: calculator("{expr}")
Observation 2: {result_2}

...

Thought N: I know the final answer.
Final Answer: {answer}
```

## 7. Tree-of-Thoughts (ToT)
```
Solve the problem by exploring multiple reasoning paths.

Problem: {problem}

Step 1: Generate K=3 distinct first-thoughts.
  Thought 1a: ...
  Thought 1b: ...
  Thought 1c: ...

Step 2: Evaluate each thought (score 0-10) for promise.
  1a: 8/10 — addresses root cause
  1b: 5/10 — partial
  1c: 3/10 — incorrect assumption

Step 3: Expand top-2 thoughts recursively.

Step 4: Return best leaf node's answer.
```

## 8. Role-Play
```
You are Dr. Chen, a senior oncologist with 20 years of experience at
Memorial Sloan Kettering. You explain medical concepts to patients
with empathy, using plain language and analogies. You always cite
sources and recommend consulting their own physician for decisions.

Patient question: {question}

Dr. Chen's response:
```

## 9. Retrieval-Augmented Generation (RAG)
```
Answer the question using ONLY the provided context. If the context
does not contain the answer, say "I don't know based on the provided context."

Context:
{retrieved_chunk_1}

{retrieved_chunk_2}

{retrieved_chunk_3}

Question: {question}

Answer (cite source IDs like [1], [2]):
```

## 10. Self-Check / Reflexion
```
Question: {question}

Initial answer: {initial_answer}

Critique your answer:
- Is each step logically sound?
- Are there unstated assumptions?
- Does it directly answer the question?

Critique: {critique}

Revised answer: {revised_answer}
```
'''


_BEST_PRACTICES = """# Prompt Engineering Best Practices

## Principles
1. **Clarity > cleverness** — explicit instructions beat clever prompting.
2. **Show, don't tell** — examples beat descriptions.
3. **Constrain output** — JSON schema, regex, or fixed format reduces parsing pain.
4. **Decompose** — break complex tasks into chained simpler prompts.
5. **Verify** — add a self-check step for high-stakes outputs.

## Token Economics
- Few-shot examples: 50-200 tokens each (truncate).
- System prompt: < 500 tokens (more dilutes focus).
- Total input: aim for < 4k tokens unless retrieval is needed.

## Robustness
- Adversarial test: inject "ignore previous instructions" — should fail.
- Empty / null inputs: define behavior explicitly.
- Multilingual: declare supported languages in system prompt.
- Long-context: place instruction at END (recency bias) or both ends.

## Versioning
- Treat prompts as code: version control, A/B test, regression suite.
- Use templating: Jinja2 / Mustache — separate prompt from data.
- Log prompt+response pairs (with PII redaction) for offline eval.

## Common Pitfalls
- "Be creative" → produces inconsistent outputs. Specify axes instead.
- Mixing languages in same prompt → degrades quality.
- Asking for confidence without calibration → over-confident outputs.
- RAG without citation enforcement → hallucinated sources.
- Few-shot with wrong labels → model learns the wrong pattern.
"""
