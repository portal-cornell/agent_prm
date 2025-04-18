#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

# Default values
declare -A MODEL_PORT_MAP
MODEL_PORT_MAP["meta-llama/Llama-3.2-3B-Instruct"]=40042

# Set default port based on MODEL_NAME
PORT=${MODEL_PORT_MAP[$MODEL_NAME]:-30000}

# Track if port was manually set
PORT_SET_MANUALLY=false

# Host is optional
HOST=""

# Directory to use the model
MODEL_DIR=""
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)
            shift
            PORT=$1
            PORT_SET_MANUALLY=true
            ;;
        -m)
            shift
            MODEL_NAME=$1
            if [ "$PORT_SET_MANUALLY" = false ]; then
                PORT=${MODEL_PORT_MAP[$MODEL_NAME]:-30000}
            fi
            ;;
        -h)
            shift
            HOST=$1
            ;;
        -d)
            MODEL_DIR="/share/portal/hw575/agent_prm/_cached_models"
            ;;
    esac
    shift
done


echo "MODEL_NAME: ${MODEL_NAME}"
echo "PORT: ${PORT}"
echo "HOST: ${HOST}"
echo "MODEL_DIR: ${MODEL_DIR}"

# Determine TMUX session ID if inside a TMUX session
SESSION_ID=""
if [ -n "$TMUX" ]; then
    SESSION_ID=$(echo "$TMUX" | awk -F',' '{print $NF}')
fi

# Default GPU ID to 0 if not set
BASE_GPU_ID=${BASE_GPU_ID:-0}

# Generate a random number between 0 and 100
RANDOM_SUFFIX=$((RANDOM % 101))

# Construct base cache directory
BASE_CACHE_DIR="$HOME/.cache/outlines"
if [ -n "$SESSION_ID" ]; then
    BASE_CACHE_DIR="${BASE_CACHE_DIR}_${SESSION_ID}"
fi
BASE_CACHE_DIR="${BASE_CACHE_DIR}/${BASE_GPU_ID}_${RANDOM_SUFFIX}"

# Export environment variable
export OUTLINES_CACHE_DIR="$BASE_CACHE_DIR"
echo "OUTLINES_CACHE_DIR: $OUTLINES_CACHE_DIR"

# Construct the Python command
CMD="python -m sglang.launch_server --model-path \"${MODEL_NAME}\" --port \"${PORT}\""
if [ -n "$HOST" ]; then
    CMD+=" --host \"${HOST}\""
fi

if [ -n "$MODEL_DIR" ]; then
    CMD+=" --download-dir \"${MODEL_DIR}\""
fi

# Run the command
eval $CMD
