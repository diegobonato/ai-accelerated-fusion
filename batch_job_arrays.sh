#!/bin/bash

total_jobs=6370  #6370 train, 1365 val , 1365 test       # Total number of jobs you need to run
batch_size=100          # Number of jobs per batch
job_script="dataset_indexing_slurm.sh"  # SLURM script to submit
account="bsc21"         # Account name
queue="gp_bsccase"      # Queue name
max_pending_jobs=2   # Maximum number of pending/running jobs allowed

# Function to check active job count
check_active_jobs() {
    squeue -u "$(whoami)" | grep -c "R\|PD"
}

# Loop to submit jobs in batches
for ((start=0; start<=total_jobs; start+=batch_size)); do
    end=$((start + batch_size - 1))
    if (( end > total_jobs )); then
        end=$total_jobs  # Ensure the last batch doesn't exceed the total jobs
    fi

    # Wait for active jobs to drop below the limit
    while (( $(check_active_jobs) >= max_pending_jobs )); do
        echo "Waiting for jobs to complete. Current active/pending jobs: $(check_active_jobs)"
        sleep 30  # Check every half minute
    done

    echo "Submitting job array from $start to $end"
    sbatch -A "$account" -q "$queue" --array="$start-$end" "$job_script" train
    sleep 1  # Optional: Add a small delay to avoid overloading the scheduler
done



total_jobs=1365  #6370 train, 1365 val , 1365 test       # Total number of jobs you need to run

# Function to check active job count
check_active_jobs() {
    squeue -u "$(whoami)" | grep -c "R\|PD"
}

# Loop to submit jobs in batches
for ((start=0; start<=total_jobs; start+=batch_size)); do
    end=$((start + batch_size - 1))
    if (( end > total_jobs )); then
        end=$total_jobs  # Ensure the last batch doesn't exceed the total jobs
    fi

    # Wait for active jobs to drop below the limit
    while (( $(check_active_jobs) >= max_pending_jobs )); do
        echo "Waiting for jobs to complete. Current active/pending jobs: $(check_active_jobs)"
        sleep 30  # Check every half minute
    done

    echo "Submitting job array from $start to $end"
    sbatch -A "$account" -q "$queue" --array="$start-$end" "$job_script" val
    sleep 1  # Optional: Add a small delay to avoid overloading the scheduler
done



total_jobs=1365  #6370 train, 1365 val , 1365 test       # Total number of jobs you need to run



# Function to check active job count
check_active_jobs() {
    squeue -u "$(whoami)" | grep -c "R\|PD"
}

# Loop to submit jobs in batches
for ((start=0; start<=total_jobs; start+=batch_size)); do
    end=$((start + batch_size - 1))
    if (( end > total_jobs )); then
        end=$total_jobs  # Ensure the last batch doesn't exceed the total jobs
    fi

    # Wait for active jobs to drop below the limit
    while (( $(check_active_jobs) >= max_pending_jobs )); do
        echo "Waiting for jobs to complete. Current active/pending jobs: $(check_active_jobs)"
        sleep 30  # Check every half minute
    done

    echo "Submitting job array from $start to $end"
    sbatch -A "$account" -q "$queue" --array="$start-$end" "$job_script" test
    sleep 1  # Optional: Add a small delay to avoid overloading the scheduler
done
