# Attributions

Nexus Coder v0.3 adapts ideas and code patterns from the following open-source projects.
All credit for the original algorithms goes to their respective authors. The code in
`nexus/integrations/` is rewritten to integrate cleanly into Nexus Coder's architecture;
it is NOT a vendored copy.

## Reference Frameworks

### 1. LitGPT (Lightning AI)
- **License**: Apache 2.0
- **Source**: https://github.com/Lightning-AI/litgpt
- **What we adapted**:
  - RoPE scaling strategies (linear / NTK-aware / YaRN) → `nexus/model/rope.py`
  - FusedLinear pattern (concatenated Q/K/V projections) → `nexus/integrations/litgpt.py`
  - PyTorch SDPA backend selection → `nexus/model/flash_attention.py`
- **Original attribution**: LitGPT: Lightning AI's LLM training toolkit. Authors: Karpathy et al. (Lightning AI), 2023-2024.

### 2. LLaMA Factory (hiyouga)
- **License**: Apache 2.0
- **Source**: https://github.com/hiyouga/LlamaFactory (also https://github.com/hiyouga/LLaMA-Factory)
- **What we adapted**:
  - Dataset format converters (Alpaca / ShareGPT / ChatML / Completion → unified Nexus format) → `nexus/integrations/llamafactory.py`
  - Concept of unified dataset registry → `nexus/data/collectors/`
- **Original attribution**: LlamaFactory: Unify Fine-tuning 100+ LLMs. Author: hiyouga.

### 3. Axolotl (axolotl-ai-cloud)
- **License**: Apache 2.0
- **Source**: https://github.com/axolotl-ai-cloud/axolotl
- **What we adapted**:
  - AxolotlStyleConfig dataclass (typed training config schema) → `nexus/integrations/axolotl.py`
  - Concept of single-YAML training configuration
- **Original attribution**: Axolotl: a simple tool for fine-tuning LLMs. Authors: winglian + axolotl-ai-cloud contributors.

### 4. OpenHands
- **License**: MIT
- **Source**: https://github.com/OpenHands/OpenHands
- **What we adapted**:
  - AgentLoop pattern (planner / executor / observer / reflector) → `nexus/integrations/openhands.py`
  - Concept of structured agent loop with reflection
- **Original attribution**: OpenHands (formerly OpenDevin): an open platform for AI software developers. Authors: OpenHands contributors.

### 5. omp-gym (Dylan Tirandaz)
- **License**: MIT
- **Source**: https://github.com/dylantirandaz/omp-gym
- **What we adapted**:
  - OpenMP optimization benchmark tasks → `nexus/integrations/omp_gym.py`
  - Concept of "predict-the-optimization" eval task
- **Original attribution**: omp-gym: An OpenMP optimization gym environment. Author: Dylan Tirandaz.

## Other Attribution

### Algorithms implemented in `nexus/model/`
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). https://arxiv.org/abs/2104.09864
- **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022). https://arxiv.org/abs/2205.14135
- **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023). https://arxiv.org/abs/2307.08691
- **ALiBi**: Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ICLR 2022). https://arxiv.org/abs/2108.12409
- **Sliding Window Attention**: Beltagy et al., "Longformer: The Long-Document Transformer" (2020). https://arxiv.org/abs/2004.05150
- **YaRN**: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023). https://arxiv.org/abs/2309.00071
- **NTK-aware RoPE scaling**: bloc97, "NTK-Aware Scaled RoPE" (2023). https://www.reddit.com/r/LocalLLaMA/comments/14lzrgj/
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020). https://arxiv.org/abs/2002.05202
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019). https://arxiv.org/abs/1910.07467
- **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). https://arxiv.org/abs/2305.13245
- **MoE**: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017). https://arxiv.org/abs/1701.06538
- **Switch Transformer**: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021). https://arxiv.org/abs/2101.03961

### Datasets referenced in `configs/sources.yaml`
- **The-Stack v2**: BigCode, https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids
- **StarCoder2-data**: BigCode, https://huggingface.co/datasets/bigcode/starcoder2data
- **CodeParrot**: CodeParrot, https://huggingface.co/codeparrot
- **Wikipedia**: Wikimedia, https://huggingface.co/wikimedia/wikipedia
- **OSCAR**: https://oscar-project.org
- **UltraChat**: HuggingFaceH4, https://huggingface.co/HuggingFaceH4/ultrachat_200k
- **OpenHermes**: teknium, https://huggingface.co/teknium/OpenHermes-2.5
- **OpenOrca**: https://huggingface.co/Open-Orca/OpenOrca
- **MetaMathQA**: https://huggingface.co/meta-math/MetaMathQA
- **GSM8K**: https://huggingface.co/datasets/gsm8k
- **HumanEval**: OpenAI, https://huggingface.co/datasets/openai_humaneval
- **MBPP**: Google Research, https://huggingface.co/datasets/mbpp
- **MATH**: https://huggingface.co/datasets/competition_math
- **FineWeb**: HuggingFaceFW, https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **Open-Web-Math**: https://huggingface.co/datasets/open-web-math/open-web-math
- **Dolma**: AllenAI, https://huggingface.co/datasets/allenai/dolma
- **Pile**: EleutherAI, https://huggingface.co/datasets/EleutherAI/pile
- **C4**: Google, https://huggingface.co/datasets/c4

### Tools inspired by existing libraries
- The `Tool` and `Skill` base classes follow the OpenAI function-calling schema pattern
- Database tools wrap established client libraries (psycopg2, pymysql, redis, pymongo, etc.)
- Web tools use `requests` + `BeautifulSoup` conventions

## License

Nexus Coder is licensed under the MIT License (see [LICENSE](LICENSE)).

The adaptations from the above projects comply with their respective licenses:
- Apache 2.0 components: retain notice, state changes
- MIT components: retain copyright notice

Where algorithms are reimplemented from academic papers, the original papers
are cited in the source files.

---

*This file is part of Nexus Coder v0.3 by Hieu Louis (2026).*
