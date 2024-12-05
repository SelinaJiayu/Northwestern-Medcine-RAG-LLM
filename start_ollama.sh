#!/bin/bash
# Start the Ollama server in the background
ollama serve &

# Wait for a few seconds to ensure the server is up
sleep 5

# Pull the required models
ollama pull llama3.1:8b
ollama pull llama3.1:70b

# Keep the container running by waiting on the server process
wait
