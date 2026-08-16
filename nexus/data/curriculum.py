"""Curriculum Learning - Học theo lộ trình từ dễ đến khó."""
from __future__ import annotations

from typing import List, Dict, Any, Iterator, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class Difficulty(str, Enum):
    """Mức độ khó của samples."""
    EASY = "easy"        # Short text, simple vocabulary
    MEDIUM = "medium"    # Standard length, normal vocabulary
    HARD = "hard"        # Long text, technical, complex
    EXPERT = "expert"    # Very long, very technical, multi-step


@dataclass
class CurriculumStage:
    """Một stage trong curriculum learning."""
    name: str
    difficulty: Difficulty
    min_length: int = 0
    max_length: int = 10000
    min_quality: float = 0.5
    weight: float = 1.0  # Sampling weight
    description: str = ""
    source_filter: Optional[List[str]] = None  # Only from these sources


class CurriculumLearning:
    """Curriculum learning scheduler.
    
    Stage 1 (EASY): Short samples, basic vocabulary
    Stage 2 (MEDIUM): Standard samples
    Stage 3 (HARD): Long technical samples
    Stage 4 (EXPERT): Very long, multi-step reasoning
    
    Usage:
        curr = CurriculumLearning()
        for stage in curr.stages:
            samples = curr.get_samples_for_stage(stage, all_samples)
            train_one_epoch(model, samples)
    """
    
    DEFAULT_STAGES = [
        CurriculumStage(
            name="stage_1_basics",
            difficulty=Difficulty.EASY,
            min_length=50,
            max_length=500,
            min_quality=0.7,
            weight=1.0,
            description="Short basic text - vocabulary building",
        ),
        CurriculumStage(
            name="stage_2_standard",
            difficulty=Difficulty.MEDIUM,
            min_length=500,
            max_length=5000,
            min_quality=0.6,
            weight=1.0,
            description="Standard length text - grammar and reasoning",
        ),
        CurriculumStage(
            name="stage_3_technical",
            difficulty=Difficulty.HARD,
            min_length=5000,
            max_length=30000,
            min_quality=0.7,
            weight=0.8,
            description="Long technical content - deep understanding",
        ),
        CurriculumStage(
            name="stage_4_expert",
            difficulty=Difficulty.EXPERT,
            min_length=30000,
            max_length=100000,
            min_quality=0.8,
            weight=0.5,
            description="Expert-level multi-step reasoning",
        ),
    ]
    
    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        self.stages = stages or self.DEFAULT_STAGES
    
    def classify_sample(self, sample: Dict[str, Any]) -> Difficulty:
        """Classify sample into difficulty level."""
        text = sample.get("text", "")
        length = len(text)
        quality = sample.get("metadata", {}).get("quality", {}).get("score", 0.5)
        
        if length < 500 and quality >= 0.7:
            return Difficulty.EASY
        elif length < 5000 and quality >= 0.6:
            return Difficulty.MEDIUM
        elif length < 30000 and quality >= 0.7:
            return Difficulty.HARD
        else:
            return Difficulty.EXPERT
    
    def get_samples_for_stage(
        self,
        stage: CurriculumStage,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter samples for a specific stage."""
        result = []
        for sample in samples:
            text = sample.get("text", "")
            length = len(text)
            quality = sample.get("metadata", {}).get("quality", {}).get("score", 0.5)
            
            # Length filter
            if not (stage.min_length <= length <= stage.max_length):
                continue
            
            # Quality filter
            if quality < stage.min_quality:
                continue
            
            # Source filter
            if stage.source_filter:
                source = sample.get("source", "")
                if source not in stage.source_filter:
                    continue
            
            result.append(sample)
        
        return result
    
    def get_curriculum_schedule(
        self,
        total_steps: int,
        num_stages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate training schedule.
        
        Returns list of {stage, start_step, end_step, samples_ratio}.
        """
        num_stages = num_stages or len(self.stages)
        stages = self.stages[:num_stages]
        
        # Allocate steps to stages (more steps to harder stages)
        total_weight = sum(s.weight for s in stages)
        schedule = []
        current_step = 0
        
        for stage in stages:
            stage_steps = int(total_steps * stage.weight / total_weight)
            schedule.append({
                "stage": stage.name,
                "difficulty": stage.difficulty.value,
                "start_step": current_step,
                "end_step": current_step + stage_steps,
                "steps": stage_steps,
                "weight": stage.weight,
                "description": stage.description,
            })
            current_step += stage_steps
        
        return schedule
