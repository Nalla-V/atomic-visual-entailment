# Atomic Visual Entailment (AVE)

Code for the master's thesis *Atomic Visual Entailment*, LIACS, Leiden University.

Visual entailment asks whether an image supports, contradicts, or is neutral
towards a sentence. Hypotheses often bundle several claims about objects,
attributes, actions and relations into one sentence, which makes a single
judgement hard to trust. AVE decomposes each hypothesis into self-contained
atomic facts, compares three ways of reasoning over them with frozen
vision-language models, combines the resulting predictions into one label, and
localises the visual evidence behind that label without region-level
supervision.

| Stage | What it does |
|---|---|
| Decomposition | hypothesis to atomic facts |
| Prediction | VLM judgements over the full hypothesis and the atomic facts |
| Selection | combine candidate predictions into one label |
| Grounding | localise the evidence for the selected prediction |
| Evaluation | accuracy, ablations, grounding recall |

## Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Repository layout](#repository-layout)
* [Run](#run)

## Installation

```bash
git clone https://github.com/Nalla-V/atomic-visual-entailment.git
cd atomic-visual-entailment
cp env.example.sh env.sh
```

Edit `env.sh` with your own paths. It is not tracked by git, so no filepaths need
to be edited anywhere else.

| Variable | Meaning |
|---|---|
| `DATA_ROOT` | directory holding `Input/` and `Output/` |
| `REPO_ROOT` | where this repository is checked out |
| `ENV_DECOMPOSE` | conda environment name for the decomposition stage |
| `HF_TOKEN` | HuggingFace token, needed only for gated models |
| `HF_HOME` | model cache directory |

Each stage has its own environment, because the vision-language models and the
text models need different library versions.

```bash
conda create -n decompose_env python=3.10 -c conda-forge -y
conda activate decompose_env
pip install -r envs/decompose.txt
```

## Dataset

Experiments use SNLI-VE, built on Flickr30k images. Grounding is evaluated
against Flickr30k Entities. Please refer to their pages for how to download the
data; the Flickr30k images require a signed request form and cannot be
redistributed here.

| Dataset | Source |
|---|---|
| SNLI-VE | https://github.com/necla-ml/SNLI-VE |
| Flickr30k images | https://shannon.cs.illinois.edu/DenotationGraph/ |
| Flickr30k Entities | https://github.com/BryanPlummer/flickr30k_entities |

Arrange them under `$DATA_ROOT`:

```
$DATA_ROOT/
├── Input/
│   ├── snli_ve_train.jsonl
│   ├── snli_ve_dev.jsonl
│   ├── snli_ve_test.jsonl
│   ├── demons.json              (included in this repository)
│   ├── flickr30k_images/        (needed from the prediction stage on)
│   └── flickr30k_entities/      (grounding evaluation only)
└── Output/                      (created automatically)
```

## Repository layout

```
src/
├── config.py                    paths, model registry, shared settings
└── decomposition/
    ├── decompose.py             hypothesis to atomic facts
    └── prompts.py               decomposition prompt

jobs/job_decompose.sh            SLURM job for the decomposition stage
envs/decompose.txt               pip requirements for the decomposition stage
env.example.sh                   template for env.sh
```

## Run

### Decomposition

Splits each hypothesis into atomic facts, using few-shot prompting with
BM25-retrieved examples from `demons.json`.

```bash
mkdir -p Logs
sbatch jobs/job_decompose.sh MODEL SPLIT [LIMIT]
```

Arguments are positional:

```bash
sbatch jobs/job_decompose.sh qwen3 dev 10        # quick check
sbatch jobs/job_decompose.sh llama dev           # full dev split
sbatch --partition=gpu-a100-80g --mem=80G --gres=gpu:a100:1 --time=3-00:00:00 \
       jobs/job_decompose.sh qwen32 train        # main run
```

The job defaults to `gpu-short` with 48 GB, which suits the 8B models. Override
partition and memory for `qwen32` as shown. `Logs/` must exist before submitting.

Without SLURM:

```bash
source env.sh
conda activate $ENV_DECOMPOSE
python -m src.decomposition.decompose --model qwen32 --split train
```

| Flag | Values | Default |
|---|---|---|
| `--model` | `qwen32`, `qwen3`, `llama` | `qwen32` |
| `--split` | `train`, `dev`, `test` | `train` |
| `--limit` | integer | all records |

| Model | Hub ID | GPU | Role |
|---|---|---|---|
| `qwen32` | `Qwen/Qwen2.5-32B-Instruct` | 80 GB | reported results |
| `qwen3` | `Qwen/Qwen3-8B` | 24 GB | decomposer comparison |
| `llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 24 GB | decomposer comparison |

Weights download from the HuggingFace Hub into `$HF_HOME` on first use. Llama is
gated: set `HF_TOKEN` and accept the licence on its model page.

Output goes to `$DATA_ROOT/Output/decompose_atoms_<model>_<split>.jsonl`, with a
matching `_debug` file holding the raw model output. Decoding is greedy, so
repeated runs give identical output. Runs append and resume from the last
completed record, so a job that hits its walltime can be resubmitted unchanged.
