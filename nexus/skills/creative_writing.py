"""Creative Writing Skill - Story / poem / fiction narrative framework.

Sinh creative writing framework: story structure templates (3-act, hero's journey,
save-the-cat beats), character & world-building worksheets, prose style guides,
revision checklist, và prompt templates cho LLM-assisted drafting.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


STORY_STRUCTURES = """
Story Structures / Cấu trúc câu chuyện
=======================================

1. Three-Act Structure / Cấu trúc 3 hồi
   - Act I  (Setup, 25%): ordinary world, inciting incident, plot point 1.
   - Act II (Confrontation, 50%): rising action, midpoint reversal, all-is-lost.
   - Act III (Resolution, 25%): climax, dénouement.

2. Hero's Journey (Campbell) / Hành trình anh hùng (12 stages)
   Ordinary World → Call to Adventure → Refusal → Mentor → Threshold →
   Tests/Allies/Enemies → Approach → Ordeal → Reward → Road Back →
   Resurrection → Return with Elixir.

3. Save the Cat! (Blake Snyder — 15 beats)
   Opening Image → Theme Stated → Set-up → Catalyst → Debate →
   Break Into Two → B Story → Fun & Games → Midpoint → Bad Guys Close In →
   All Is Lost → Dark Night of Soul → Break Into Three → Finale → Final Image.

4. Seven-Point Story Structure (Dan Wells)
   Hook → Plot Turn 1 → Pinch 1 → Midpoint → Pinch 2 → Plot Turn 2 → Resolution.

5. Kishōtenketsu (East-Asian narrative without conflict)
   Ki (Introduction) → Shō (Development) → Ten (Twist) → Ketsu (Conclusion).

Selecting a structure:
  - Western audience / conflict-driven → 3-act or Save-the-Cat.
  - Mythic scope → Hero's Journey.
  - Short story / vignette → Kishōtenketsu or Seven-Point.
"""

CHARACTER_WORKSHEET = """# Character Worksheet / Hồ sơ nhân vật

## Identity / Bản sắc
- Full name:
- Nickname / alias:
- Age, gender, pronouns:
- Occupation:

## Physical / Ngoại hình
- Hair, eyes, build, distinguishing marks:

## Voice / Giọng văn
- Speech pattern / catchphrases:
- Education level (affects vocabulary):
- Cultural background (idioms, proverbs):

## Psychology / Tâm lý
- Core desire (conscious):
- Core need (subconscious):
- Ghost / wound (past trauma):
- Flaw (the trait that must be overcome):
- Strength (the trait that will save them):

Arc / Sự phát triển nhân vật
- Beginning state (LIE they believe):
- Inciting moment that cracks the lie:
- Midpoint realization (truth vs lie):
- Climax choice that proves growth:
- Ending state (truth they now live):

Relationships / Mối quan hệ
- Allies:
- Antagonists:
- Mentors:
- Foils:
"""

WORLDBUILDING = """
World-Building Checklist / Checklist xây thế giới
==================================================
[ ] Cosmology & geography — maps, climate, seasons
[ ] History — timeline of major events
[ ] Politics — government, factions, conflicts
[ ] Economy — currency, trade, scarcity
[ ] Religion & belief systems — rituals, afterlife
[ ] Magic / technology system — rules, costs, limits
[ ] Language & naming conventions — phonology, scripts
[ ] Culture — food, clothing, art, music, festivals
[ ] Daily life — what a normal day looks like
[ ] Taboos & laws — what's forbidden, what's enforced

Show, don't tell: reveal world through character interactions,
not info-dumps. Trust readers to infer from context.
"""

PROSE_STYLE = """
Prose Style Guide / Hướng dẫn văn phong
=========================================
- Strong verbs over adverbs: "sprinted" not "ran quickly"
- Concrete sensory detail: not "the house was creepy" — "the wallpaper
  peeled in long, yellow strips; something smelled of mold and old paper."
- Cut filter words: "she saw", "he felt", "I noticed" — narrate directly.
- Vary sentence rhythm: short for tension, long for immersion.
- Dialogue: every line should reveal character or advance plot (ideally both).
- Metaphors: one strong image > three weak ones. Avoid cliché.
- Active voice by default; passive for effect (mystery, victimization).

Revision Checklist / Checklist chỉnh sửa
  [ ] Open with a hook (within first 2 sentences)
  [ ] Each scene has goal → conflict → disaster or decision
  [ ] Stakes escalate — does each scene top the previous?
  [ ] Show character growth, not just external plot
  [ ] Ending resonates with theme stated early
  [ ] Cut 10% of words (Stephen King's rule)
  [ ] Read aloud — does it flow? Cut what trips you up.
"""

PROMPT_TEMPLATES = """# Creative Writing Prompt Templates / Mẫu prompt sáng tác

## Story Generation
As a [genre] author in the style of [writer], write a [length] story
about [protagonist] who wants [desire] but faces [obstacle].
Setting: [world]. Tone: [mood].
Theme to weave in: [theme].
Begin in media res. End with a resonant final image.

## Character Sketch
Create a character using this profile:
- Desire:
- Wound:
- Flaw:
- Secret:
Write a 300-word scene showing them under pressure, revealing
their flaw and hinting at their wound without stating it.

## Poem (form-locked)
Write a [sonnet | haiku | villanelle | ghazal] on the theme of [topic].
Meter: [iambic pentameter | free verse].
Rhyme scheme: [abab cdcd | none].
Use at least one striking metaphor that surprises but fits.

## Scene Revision
Revise this scene to:
- Cut filter words ("saw", "felt", "noticed")
- Replace 3 adverbs with stronger verbs
- Add 2 sensory details (smell, touch, sound)
- Sharpen the dialogue — remove "said" tags where action can do the work

Original scene:
[paste scene]

## World-Building
Design a [fantasy | sci-fi | alt-history] setting where [what-if premise].
Return: cosmology (3 sentences), politics (3 factions), magic/tech system
(rules + costs), and one daily-life vignette (200 words).
"""


class CreativeWritingSkill(Skill):
    """Sinh framework sáng tác: story structures, character, world, prompts."""

    category = SkillCategory.LANGUAGE
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "write story", "write poem", "creative", "creative writing",
        "fiction", "narrative", "novel", "short story",
        "character", "worldbuilding", "world-building", "prose",
    ]
    examples = [
        "Viết truyện ngắn theo cấu trúc 3 hồi",
        "Tạo nhân vật với worksheet chi tiết",
        "Sinh prompt sáng tác thơ",
    ]

    @property
    def name(self) -> str:
        return "creative_writing"

    @property
    def description(self) -> str:
        return (
            "Sinh framework sáng tác: story structures (3-act, hero's journey, "
            "save-the-cat, kishōtenketsu), character & world-building worksheets, "
            "prose style guide, và LLM prompt templates cho drafting."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "poem" in prompt_lower or "poetry" in prompt_lower:
            focus = "poetry"
        elif "character" in prompt_lower:
            focus = "character"
        elif "world" in prompt_lower:
            focus = "worldbuilding"
        else:
            focus = "story"

        artifacts: List[Dict[str, str]] = [
            {"name": "STORY_STRUCTURES.md", "language": "markdown", "content": STORY_STRUCTURES},
            {"name": "CHARACTER_WORKSHEET.md", "language": "markdown", "content": CHARACTER_WORKSHEET},
            {"name": "WORLDBUILDING.md", "language": "markdown", "content": WORLDBUILDING},
            {"name": "PROSE_STYLE.md", "language": "markdown", "content": PROSE_STYLE},
            {"name": "PROMPT_TEMPLATES.md", "language": "markdown", "content": PROMPT_TEMPLATES},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[creative_writing] focus={focus}\n"
                f"Generated story structures (5), character & world-building "
                f"worksheets, prose style guide, revision checklist, and LLM "
                f"prompt templates."
            ),
            artifacts=artifacts,
            suggestions=[
                "Pick ONE structure first — don't mix 3-act with hero's journey mid-draft",
                "Write character's lie vs truth before drafting the story arc",
                "Use 'show, don't tell' — replace adjectives with sensory details",
                "Cut 10% of words in revision (Stephen King's rule)",
                "Read dialogue aloud to catch unnatural phrasing",
                "Begin in media res — open with action, not backstory",
            ],
            metadata={
                "skill": self.name,
                "focus": focus,
                "structures_covered": [
                    "three_act", "heros_journey", "save_the_cat",
                    "seven_point", "kishotenketsu",
                ],
                "deliverables": [
                    "story_structures", "character_worksheet",
                    "worldbuilding", "prose_style", "prompt_templates",
                ],
                "version": self.version,
                "author": self.author,
            },
        )
