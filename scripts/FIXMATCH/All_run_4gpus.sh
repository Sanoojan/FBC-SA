#!/bin/bash

# List of tasks with each line in the format "TASK_NAME ARG1 ARG2 ARG3_PLACEHOLDER ARG4"
TASKS=(
  "ssdg_officehome 975 0 Baseline"
  "ssdg_officehome 1950 0 Baseline"
  "ssdg_pacs 210 0 Baseline"
  "ssdg_pacs 105 0 Baseline"
)

# GPUs to be used
GPU_IDS=(0 1 2 3)

# Track GPU usage
declare -A gpu_status

# Function to check if a GPU is free based on process memory usage
is_gpu_free() {
  local GPU_ID=$1
  # Check if there is any process using significant memory on the GPU
  local MEM_USAGE=$(nvidia-smi --id=$GPU_ID --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
  if [[ -z $MEM_USAGE || $MEM_USAGE -lt 500 ]]; then
    return 0
  else
    return 1
  fi
}

# Function to run a command on a specific GPU
run_task() {
  local GPU_ID=$1
  TASK=($2)  # Convert string to array
  TASK[2]=$GPU_ID  # Set the GPU ID dynamically
  CUDA_VISIBLE_DEVICES=$GPU_ID bash run_ssdg_dummy.sh "${TASK[@]}" &
  echo $!  # Return the PID of the process
}

# Function to update GPU status
update_gpu_status() {
  for GPU_ID in "${GPU_IDS[@]}"; do
    if is_gpu_free $GPU_ID; then
      gpu_status[$GPU_ID]="free"
    else
      gpu_status[$GPU_ID]="busy"
    fi
  done
}

# Function to update running tasks
update_running_tasks() {
  for PID in "${!running_tasks[@]}"; do
    if ! ps -p $PID > /dev/null; then
      local GPU_ID=${running_tasks[$PID]}
      gpu_status[$GPU_ID]="free"
      unset running_tasks[$PID]
    fi
  done
}

# Initialize GPU status
for GPU_ID in "${GPU_IDS[@]}"; do
  gpu_status[$GPU_ID]="free"
done

# Main loop to manage task allocation
while [[ ${#TASKS[@]} -gt 0 || ${#running_tasks[@]} -gt 0 ]]; do
  update_running_tasks
  update_gpu_status

  for GPU_ID in "${GPU_IDS[@]}"; do
    if [[ ${#TASKS[@]} -eq 0 ]]; then
      break
    fi

    if [[ ${gpu_status[$GPU_ID]} == "free" ]]; then
      TASK="${TASKS[0]}"
      TASKS=("${TASKS[@]:1}")
      PID=$(run_task $GPU_ID "$TASK")
      running_tasks[$PID]=$GPU_ID
      gpu_status[$GPU_ID]="busy"
      echo "Started task on GPU $GPU_ID: $TASK"
    else
      echo "GPU $GPU_ID is busy"
    fi
  done

  sleep 5  # Brief sleep to prevent tight loop
done

wait  # Wait for all tasks to complete before exiting