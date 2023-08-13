#!/bin/bash

echo "Starting front end"
python3 -m http.server --directory ./spa/dist/ 3000 &
pid1=$!

echo "Starting back end"
uvicorn app:app
pid2=$!

wait $pid1
wait $pid2