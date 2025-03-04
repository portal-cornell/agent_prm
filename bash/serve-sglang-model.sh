#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"

# Default values
declare -A MODEL_PORT_MAP
MODEL_PORT_MAP["meta-llama/Llama-3.2-3B-Instruct"]=40042

# Set default port based on MODEL_NAME
PORT=${MODEL_PORT_MAP[$MODEL_NAME]:-30000}

# Track if port was manually set
PORT_SET_MANUALLY=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p)
            shift
            PORT=$1
            PORT_SET_MANUALLY=true  # Mark port as manually set
            ;;
        -m)
            shift
            MODEL_NAME=$1
            # Only update PORT from mapping if it hasn't been manually set
            if [ "$PORT_SET_MANUALLY" = false ]; then
                PORT=${MODEL_PORT_MAP[$MODEL_NAME]:-30000}
            fi
            ;;
    esac
    shift
done

python -m sglang.launch_server --model-path "${MODEL_NAME}" --port "${PORT}"
