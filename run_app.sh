#!/bin/bash

# Function to gracefully terminate the processes
cleanup() {
    echo "Stopping processes..."
    kill -SIGTERM $pid1
    kill -SIGTERM $pid2
    wait $pid1
    wait $pid2
    echo "All processes stopped."
    exit 0
}

# Trap termination signals and call the cleanup function
trap cleanup SIGINT SIGTERM

python3 -m http.server --directory ./spa/dist/ 3000 &
pid1=$!
echo "Front end started at pid" $pid1

uvicorn --host 0.0.0.0 app:app
pid2=$!
echo "Back end started at pid" $pid2

wait $pid1
wait $pid2
