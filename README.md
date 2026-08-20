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
| `ENV_PREDICT` | conda environment name for the prediction stage |
| `ENV_SELECTION` | conda environment name for the selection stage |
| `ENV_GROUNDING` | conda environment name for the grounding stage |
| `HF_TOKEN` | HuggingFace token, needed only for gated models |
| `HF_HOME` | model cache directory |

Each stage has its own environment, because the vision-language models and the
text models need different library versions.

```bash
conda create -n decompose_env python=3.10 -c conda-forge -y
conda activate decompose_env
pip install -r envs/decompose.txt

conda create -n predict_env python=3.10 -c conda-forge -y
conda activate predict_env
pip install -r envs/predict.txt

conda create -n selection_env python=3.10 -c conda-forge -y
conda activate selection_env
pip install -r envs/selection.txt

conda create -n grounding_env python=3.10 -c conda-forge -y
conda activate grounding_env
pip install -r envs/grounding.txt
```

The selection stage runs on CPU only, so `envs/selection.txt` has no CUDA
dependencies.

The environments are not interchangeable. Decomposition, prediction and
grounding phrase extraction need `transformers` 4.x, while Grounding DINO
needs 5.x: its image processor changed between the two major versions, and
using the wrong one shifts every predicted box by a few pixels without
raising an error.

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
├── decomposition/
│   ├── decompose.py             hypothesis to atomic facts
│   └── prompts.py               decomposition prompt
├── prediction/
│   ├── predict.py               entry point, resume, output writing
│   ├── models.py                VLM registry and per-family adapters
│   ├── common.py                parsing, label scoring, confidence diagnostics
│   ├── prompts.py               prediction prompts
│   ├── baseline.py              full-hypothesis prediction
│   ├── joint.py                 joint atomic prediction
│   ├── selfdecompose.py         self-decomposition prediction
│   └── independent.py           independent atomic prediction
├── selection/
│   ├── voting.py                majority voting over the candidate pool
│   ├── train.py                 learned selector training and model selection
│   ├── evaluate.py              learned selector evaluation on dev and test
│   ├── analysis.py              ablations, importance, and the figures
│   ├── features.py              meta-feature extraction
│   ├── classifiers.py           classifier configurations
│   ├── candidates.py            the K=12 candidate pool
│   └── common.py                metrics and shared helpers
└── grounding/
    ├── phrases.py               reasoning to groundable phrases
    ├── detect.py                Grounding DINO boxes and figures
    ├── summary.py               Flickr30k Entities evaluation
    ├── prompts.py               phrase extraction prompt
    └── common.py                phrase cleaning helpers

jobs/                            SLURM jobs, one per stage
envs/                            pip requirements, one per stage
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

### Prediction

Runs the VLMs over each image with the full hypothesis and with the atomic facts
from the decomposition stage. Both prompt styles, simple and structured, run
together from a single model load.

```bash
sbatch jobs/job_predict.sh VLM SPLIT METHOD [LIMIT]
```

Arguments are positional:

```bash
sbatch jobs/job_predict.sh internvl dev all 10   # quick check, all methods
sbatch jobs/job_predict.sh internvl test all     # full test split
sbatch jobs/job_predict.sh qwen3 test joint      # one method
sbatch --partition=gpu-a100-80g --mem=80G --gres=gpu:a100:1 \
       jobs/job_predict.sh qwen3vl_32b dev all   # 32B model
```

Without SLURM:

```bash
source env.sh
conda activate $ENV_PREDICT
python -m src.prediction.predict --vlm internvl --split test --method all
```

| Flag | Values | Default |
|---|---|---|
| `--vlm` | see table below | required |
| `--split` | `train`, `dev`, `test` | required |
| `--method` | `baseline`, `joint`, `selfdecompose`, `independent`, `all` | required |
| `--limit` | integer | all records |
| `--decomposer` | decomposer whose atoms to read | `qwen32` |

| VLM | Hub ID | Precision | Methods |
|---|---|---|---|
| `qwen3` | `Qwen/Qwen3-VL-8B-Instruct` | bfloat16 | all four |
| `internvl` | `OpenGVLab/InternVL3-8B-hf` | bfloat16 | all four |
| `qwen3vl_32b` | `Qwen/Qwen3-VL-32B-Instruct` | bfloat16 | baseline |
| `llava` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | float16 | baseline |
| `idefics2` | `HuggingFaceM4/idefics2-8b` | float16 | baseline |
| `qwen2vl_2b` | `Qwen/Qwen2-VL-2B-Instruct` | bfloat16 | baseline |
| `internvl3_1b` | `OpenGVLab/InternVL3-1B-hf` | bfloat16 | baseline |

`qwen3` and `internvl` form the AVE prediction pool and run all four prediction
methods. The other five appear in the VLM comparison, which uses full-hypothesis
prediction, so `all` resolves to `baseline` for them.

| Method | What the VLM sees |
|---|---|
| `baseline` | the full hypothesis |
| `joint` | the atomic facts together |
| `selfdecompose` | the hypothesis, decomposed by the VLM itself |
| `independent` | one atomic fact at a time |

Output goes to `$DATA_ROOT/Output/<vlm>_predictions/<method>_<prompt>_<split>.jsonl`,
with a matching `_debug` file. Runs resume by input line index, so a job that
hits its walltime can be resubmitted unchanged.

### Selection

Combines the candidate predictions into one label. Both approaches read the
prediction files written by the previous stage and run on CPU.

#### Majority voting

```bash
sbatch jobs/job_voting.sh
```

Reports the best individual candidates, intra-model voting per VLM (K=6),
inter-model voting over the full pool (K=12), and the oracle upper bound.
Output goes to `$DATA_ROOT/Output/majority_voting_summary/`.

#### Learned selection

Training compares four feature variants against seven classifier
configurations on one fixed stratified split, then saves the best model per
variant.

```bash
sbatch jobs/job_train_selector.sh [OUTPUT_NAME]
```

| Flag | Meaning | Default |
|---|---|---|
| `--output-name` | output folder under `Output/` | `AVE_train_learned_selector_v3` |
| `--validation-size` | held-out fraction of the training pool | `0.20` |
| `--random-state` | seed for the split and the classifiers | `42` |

| Feature variant | Candidate pool |
|---|---|
| `baseline_only` | full-hypothesis only, K=4 |
| `simple_atomic` | atomic methods, simple prompt, K=4 |
| `structured_atomic` | atomic methods, structured prompt, K=4 |
| `full_12_methods` | 2 VLMs x 3 methods x 2 prompts, K=12 |

Evaluation loads the saved model and applies it to a split:

```bash
sbatch jobs/job_evaluate_selector.sh dev [TRAIN_RUN_NAME]
sbatch jobs/job_evaluate_selector.sh test [TRAIN_RUN_NAME]
```

The post-training analyses are separate, since they take considerably longer
than training itself:

```bash
sbatch jobs/job_analysis.sh [OUTPUT_NAME]
```

This runs the data-efficiency sweep, permutation importance, and the
selector-objective ablation, and writes the figures. Use
`--skip data_efficiency importance objective` to redraw the figures from the
saved CSVs without recomputing.

Training is seeded, so a rerun reproduces the same model and the same
validation metrics.

### Grounding

Links the selected prediction to visible image regions. Grounding applies only
to entailment and contradiction predictions, and only when the selected
candidate agrees with the final label; the evaluation stage marks which rows
qualify.

Three steps, in order.

#### Phrase extraction

Qwen3-8B turns each piece of evidence from the selected candidate's reasoning
into a short visual phrase, plus a shorter match phrase used later for
Flickr30k Entities matching.

```bash
sbatch jobs/job_grounding_phrases.sh SPLIT [LIMIT]
```

Extraction stops once each groundable label reaches `--target-per-label`,
2500 by default, so grounding runs on a sample rather than the whole split.

#### Detection

```bash
sbatch jobs/job_grounding_detect.sh SPLIT [LIMIT]
```

Grounding DINO localises each phrase. Near-duplicate boxes are suppressed and
the top three are kept, which is what makes Recall@3 and Mean IoU@3 meaningful.
Two figures are written per row: a diagnostic panel showing the hypothesis,
atoms, phrases and reasoning beside the boxed image, and a report-ready image
with the boxes alone. Pass `--no-images` to skip both.

| Setting | Value |
|---|---|
| Model | `IDEA-Research/grounding-dino-base` |
| Box threshold | 0.35 |
| Text threshold | 0.25 |
| Boxes kept | 3 |

#### Evaluation

```bash
sbatch jobs/job_grounding_summary.sh SPLIT
```

Matches each phrase to human-annotated Flickr30k Entities phrases, then reports
coverage, Recall@1, Recall@3 and mean best IoU@3 over matched rows, split by
predicted label. Reads `annotations.zip` directly, so the archive does not need
unpacking.

Without SLURM:

```bash
source env.sh
conda activate $ENV_GROUNDING
python -m src.grounding.phrases --split test
python -m src.grounding.detect --split test
conda activate $ENV_SELECTION
python -m src.grounding.summary --split test
```

Phrase extraction uses the decomposition environment, detection uses the
grounding environment, and the evaluation needs only the standard library.
