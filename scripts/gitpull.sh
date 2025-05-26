#!/bin/bash

# Infinite loop to keep running the command periodically
while true; do
    # Run the git pull command
    git pull
    echo "Executed git pull at $(date)"
    # Sleep for 0.5 minutes (30 seconds)
    sleep 30
done
