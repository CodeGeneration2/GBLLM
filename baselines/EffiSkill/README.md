# EffiSkill: ASE 2026 Artifact

This directory contains the artifact for the ASE 2026 submission:

`Mining Agent Skills for Automated Code Efficiency Optimization`

All commands and paths in this README are relative to this directory.

## Overview

The artifact supports reproduction of the main paper outputs:

- Table 1
- Figures 4, 5, and 6
- Table 3

The artifact also includes the released Python and C++ pipeline code used in the study. Reviewers who have access to an OpenAI-compatible endpoint may run small live pipeline examples in addition to the paper reproduction scripts.

## Directory Structure

- `src/effiskill_artifact/`
  - command-line interface
  - paper reproduction scripts
  - Python and C++ pipeline implementation
- `data/mining/`
  - mining pairs for Python and C++
- `data/rq1/evaluation/`
  - evaluation results used for Table 1 and Figures 4 to 6
- `data/rq4/`
  - skill-analysis inputs used for Table 3 and the RQ4 statistics
- `outputs/`
  - reference outputs generated from the packaged inputs

## Installation

Recommended environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installation is sufficient for:

- `reproduce-rq1`
- `reproduce-rq4`
- `reproduce-paper`
- `build-registry`

To install the full dependency set for the live Stage I / Stage II pipeline:

```bash
python -m pip install -r requirements-full.txt
```

## Reproducing Paper Results

To reproduce all packaged paper-facing outputs:

```bash
effiskill-artifact reproduce-paper
```

This command generates:

- `outputs/rq1/table1_rq1.csv`
- `outputs/rq1/table1_rq1.tex`
- `outputs/rq1/figure4_topk_growth.pdf`
- `outputs/rq1/figure5_task_level_comparison.pdf`
- `outputs/rq1/figure6_top8_bucket_distribution.pdf`
- `outputs/rq4/table3_skill_family_summary.csv`
- `outputs/rq4/table3_skill_family_summary.tex`
- `outputs/rq4/rq4_claims.csv`

The two parts may also be reproduced separately:

```bash
effiskill-artifact reproduce-rq1
effiskill-artifact reproduce-rq4
```

## Optional Pipeline Execution

The live pipeline requires an OpenAI-compatible endpoint. Reviewers who wish to run the pipeline should provide:

```bash
export EFFISKILL_BASE_URL="https://your-openai-compatible-endpoint/v1"
export EFFISKILL_API_KEY="your-api-key"
export EFFISKILL_MODEL="gpt-5.1"
export EFFISKILL_LLM_TIMEOUT_SEC="180"
```

The same values may also be passed through command-line arguments such as `--base-url`, `--api-key`, and `--model`.

Example Python workflow:

```bash
effiskill-artifact pairs-to-traces --language python -- --input data/mining/python/python_ab_test.jsonl --output outputs/demo_python_traces.jsonl --cache-dir outputs/cache_python --fail-dir outputs/fail_python --model "$EFFISKILL_MODEL"
effiskill-artifact extract-skills --language python -- --gen-dir outputs/demo_python_library --traces outputs/demo_python_traces.jsonl --model "$EFFISKILL_MODEL"
effiskill-artifact build-registry --language python -- --skills_dir outputs/demo_python_library/skills --out outputs/demo_python_library/skills/registry.json
```

Example C++ workflow:

```bash
effiskill-artifact pairs-to-traces --language cpp -- --input data/mining/cpp/cpp_ab_test.jsonl --output outputs/demo_cpp_traces.jsonl --cache-dir outputs/cache_cpp --fail-dir outputs/fail_cpp --model "$EFFISKILL_MODEL"
effiskill-artifact extract-skills --language cpp -- --gen-dir outputs/demo_cpp_library --traces outputs/demo_cpp_traces.jsonl --model "$EFFISKILL_MODEL"
effiskill-artifact build-registry --language cpp -- --skills_dir outputs/demo_cpp_library/skills --out outputs/demo_cpp_library/skills/registry.json
```

Example inference command:

```bash
effiskill-artifact infer --language python -- --dataset custom --data your_tasks.csv --slow-column slow_code --skills-root outputs/demo_python_library/skills --out outputs/demo_python_runs.jsonl --model "$EFFISKILL_MODEL"
```

## Notes

- The `outputs/` directory may be deleted and regenerated using the packaged scripts and inputs.
- Command-line help is available through `effiskill-artifact --help`.




## Note

The baselines folder is mostly derived from the paper "EFFI-LEARNER: Enhancing Efficiency of Generated Code via Self-Optimization", with its Github URL: https://doi.org/10.5281/zenodo.19249527