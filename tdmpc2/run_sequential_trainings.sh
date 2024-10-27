#!/bin/bash

# Set number of training steps in each run
TRAINING_STEPS=2_000_000
#TRAINING_STEPS=100


#PROJECT_NAME="MacBook_Testrun"
PROJECT_NAME="Evaluation_Run_DFKI"


# List of seed values
SEEDS=(0 1 2 3 4 5)
# Loop through each script
for seed in  "${SEEDS[@]}" ; do
    echo "Starting walk $seed..."

    # Run the walk script and capture its exit status
    python -m tdmpc2.train disable_wandb=False wandb_entity=qqw exp_name=my_reward task=humanoid_h1-walk-v0 seed=$seed steps=$TRAINING_STEPS wandb_project=${PROJECT_NAME}_walk_$seed log_everything=true eval_freq=10_000 eval_episodes=1

    EXIT_STATUS=$?
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "$seed failed with exit code $EXIT_STATUS. Continuing to the next script."
    else
        echo "$seed completed successfully."
    fi

    # Pause for cleanup
    echo "Sleeping for $SLEEP_DURATION seconds before the next run..."
    sleep 5

     echo "Starting push $seed..."

     # Run the üush script and capture its exit status
    python -m tdmpc2.train disable_wandb=False wandb_entity=qqw exp_name=my_reward task=humanoid_h1-push-v0 seed=$seed steps=$TRAINING_STEPS wandb_project=${PROJECT_NAME}_push_$seed log_everything=true eval_freq=10_000 eval_episodes=1

    EXIT_STATUS=$?
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "$seed failed with exit code $EXIT_STATUS. Continuing to the next script."
    else
        echo "$seed completed successfully."
    fi

    # Pause for cleanup
    echo "Sleeping for $SLEEP_DURATION seconds before the next run..."
    sleep $SLEEP_DURATION
done

echo "All scripts finished running."
