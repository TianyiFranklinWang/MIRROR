#!/bin/bash
set -e  # Exit immediately on error

# Function to display usage information
usage() {
    echo "Usage: $0 <nnodes> <nproc_per_node> <rdzv_backend> <rdzv_endpoint> <config_file> <fold_nb> [additional_args...]"
    exit 1
}

# Ensure minimum required arguments
if [ "$#" -lt 6 ]; then
    usage
fi

# Assign required arguments
nnodes="$1"
nproc_per_node="$2"
rdzv_backend="$3"
rdzv_endpoint="$4"
config_file="$5"
fold_nb="$6"
shift 6  # Remove first six arguments

# Extract additional arguments (must start with "--")
additional_args=()
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == --* ]]; then
        if [[ "$#" -gt 1 && "$2" != --* ]]; then
            additional_args+=("$1" "$2")  # Store key-value pair
            shift 2  # Consume key-value
        else
            additional_args+=("$1")  # Store standalone flag
            shift 1
        fi
    else
        echo "Error: Unexpected positional argument '$1'. Additional arguments must begin with '--'."
        usage
    fi
done

# Validate that nnodes and nproc_per_node are positive integers
if ! [[ "$nnodes" =~ ^[0-9]+$ ]]; then
    echo "Error: <nnodes> must be a positive integer."
    usage
fi

if ! [[ "$nproc_per_node" =~ ^[0-9]+$ ]]; then
    echo "Error: <nproc_per_node> must be a positive integer."
    usage
fi

# Check if the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' not found."
    exit 1
fi

# Set environment variables
export OMP_NUM_THREADS=16

# Construct the torchrun command as an array
cmd=(
    torchrun
    --nnodes "$nnodes"
    --nproc_per_node "$nproc_per_node"
    --rdzv_backend "$rdzv_backend"
    --rdzv_endpoint "$rdzv_endpoint"
    train_mirror.py
    --config "$config_file"
    --fold-nb "$fold_nb"
)

# Append additional arguments if provided
if [ "${#additional_args[@]}" -gt 0 ]; then
    cmd+=("${additional_args[@]}")
fi

# Display the command for debugging purposes
echo "Executing training command: ${cmd[*]}"

# Execute the command with error handling
if ! "${cmd[@]}"; then
    echo "Error: Training process failed. Exiting..."
    exit 1
fi

# Completion message
echo "Training completed successfully."
