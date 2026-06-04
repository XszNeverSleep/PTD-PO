# PTD-PO: Privileged Tutoring Distillation for Multimodal Policy Optimization

[![GitHub](https://img.shields.io/badge/GitHub-PTD--PO-green)](https://github.com/XszNeverSleep/PTD-PO)
[![Framework](https://img.shields.io/badge/Built%20on-EasyR1-blue)](https://github.com/hiyouga/EasyR1)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](./LICENSE)

This repository contains the official implementation of **PTD-PO**, a research fork of [EasyR1](https://github.com/hiyouga/EasyR1) for multimodal reinforcement learning. PTD introduces a privileged tutoring branch into on-policy optimization: the student samples from the standard prompt, while the teacher evaluates the same trajectory with hint-augmented context and provides token-level distillation signals.

The codebase keeps EasyR1's scalable VLM training stack and adds PTD variants for GRPO/GSPO-style training.

---

## News

- **[2026-06-04]** We released the first **PTD-PO** codebase.
- **[2026-06-04]** PTD-GRPO and PTD-GSPO example scripts are available under `examples/ptd-grpo/`.

---

## The Core Insight: Hard Rollouts Need Privileged Guidance

Outcome-only RL signals are sparse and coarse for multimodal reasoning. When a rollout group fails or contains weak trajectories, the policy receives little information about *how* the reasoning should move toward the correct visual or mathematical evidence.

PTD addresses this by adding a teacher view with privileged information. The teacher does not generate a new answer during training; instead, it scores the student's sampled trajectory under a hint-augmented prompt and turns the privileged context into dense token-level feedback.

<p align="center">
  <img src="assets/ptd_insight.png" alt="PTD insight" width="88%">
</p>

---

## Our Solution: PTD-Enhanced Policy Optimization

PTD is designed as a lightweight extension to existing on-policy RL algorithms. It keeps the rollout and reward pipeline unchanged, then adds a distillation objective on selected samples:

1. **Student rollout**: the actor samples responses from the standard prompt.
2. **Group filtering**: PTD activates on groups whose accuracy is below a configurable threshold.
3. **Privileged teacher scoring**: a frozen reference teacher evaluates the same response with a hint-augmented prompt.
4. **Token-level distillation**: the actor receives an auxiliary KL/JSD loss from the teacher distribution.

<p align="center">
  <img src="assets/ptd_framework.png" alt="PTD framework" width="92%">
</p>

PTD currently supports:

- PTD-GRPO and PTD-GSPO training scripts.
- Top-k teacher distillation with tail compensation.
- Full-vocabulary distillation for exact KL/JSD when memory allows.
- Frozen reference teacher or old-policy teacher modes.
- Hint-augmented text/VLM prompts through `prompt_with_hint`.

---

## Main Results

PTD improves multimodal reasoning by injecting dense guidance into hard rollout groups while preserving the standard on-policy optimization path.

<p align="center">
  <img src="assets/main_results.jpg" alt="Main results" width="92%">
</p>

---

## Getting Started

### 1. Environment

Recommended environment follows EasyR1:

- Python 3.9+
- `transformers>=4.54.0`
- `flash-attn>=2.4.3`
- `vllm>=0.8.3`

We recommend the EasyR1 Docker image:

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

### 2. Installation

```bash
git clone https://github.com/XszNeverSleep/PTD-PO.git
cd PTD-PO
pip install -e .
```

### 3. Training with PTD

Run a PTD-GRPO example:

```bash
bash examples/ptd-grpo/qwen3_vl_4b_thinking_ViRL39K_ref_ptd-grpo_jsd_top100_5e-1.sh
```

Run the multi-node variant from the Ray head node:

```bash
bash examples/ptd-grpo/launch_multinode.sh \
  examples/ptd-grpo/qwen3_vl_4b_thinking_ViRL39K_ref_ptd-grpo_jsd_top100_5e-1_thr_1_rollout_8.sh
```

Core PTD options use the `ptd` namespace:

```bash
algorithm.enable_ptd=true
algorithm.ptd_threshold=1.0
algorithm.ptd_top_k=100
algorithm.ptd_coef=5.0e-1
algorithm.ptd_kl_direction=jsd_kl
algorithm.ptd_use_ref_teacher=true
```

### 4. Merge Checkpoints

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

---

## Data Format

PTD uses the standard EasyR1 dataset interface plus an optional hint-augmented prompt column:

- `prompt`: student prompt.
- `answer`: ground-truth answer for reward computation.
- `images` / `videos`: optional multimodal inputs.
- `prompt_with_hint`: privileged teacher prompt used only for PTD teacher scoring.

Enable hint loading with:

```bash
data.prompt_with_hint_key=prompt_with_hint
data.max_hint_prompt_length=4096
```

---

## Repository Structure

```text
examples/ptd-grpo/        PTD-GRPO and PTD-GSPO training scripts
examples/reward_function/ Reward functions for math and VLM reasoning
verl/trainer/             RL training loop and PTD mask construction
verl/workers/actor/       Actor update, teacher scoring, and PTD loss
scripts/model_merger.py   FSDP checkpoint merger
```

---

## Acknowledgements

PTD-PO is built on [EasyR1](https://github.com/hiyouga/EasyR1), [veRL](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), and the broader open-source VLM RL ecosystem. We thank the original authors for providing a strong and extensible training foundation.

---

## Citation

Citation information will be updated with the paper release. For now, please cite this repository if you use PTD-PO:

```bibtex
@misc{ptdpo2026,
  title        = {PTD-PO: Privileged Tutoring Distillation for Multimodal Policy Optimization},
  howpublished = {\url{https://github.com/XszNeverSleep/PTD-PO}},
  year         = {2026}
}
```
