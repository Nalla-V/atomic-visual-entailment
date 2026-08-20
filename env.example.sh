#!/bin/bash
# Copy to env.sh and edit. env.sh is gitignored.

export DATA_ROOT="/path/to/data"      # holds Input/ and Output/
export REPO_ROOT="/path/to/atomic-visual-entailment"

export ENV_DECOMPOSE="decompose_env"     # decomposition, selection
export ENV_PREDICT="predict_env"         # VLM prediction
export ENV_SELECTION="selection_env"     # selection
export ENV_GROUNDING="grounding_env"     # grounding

export HF_TOKEN=""                    # only for gated models (Llama)
export HF_HOME="$DATA_ROOT/.hf_cache" 