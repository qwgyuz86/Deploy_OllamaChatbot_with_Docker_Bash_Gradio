#!/bin/bash

# Step 0: Create a Custom Docker Network (if not already created)
echo "Creating Docker network..."
docker network create chatbot-net || true  # Avoid errors if the network already exists

# Step 1: Start Ollama Container
echo "Starting Ollama container..."
docker run -d --network chatbot-net -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Wait for Ollama to start
echo "Waiting for Ollama to initialize..."
sleep 10

# Step 2: Load Llama Model in Ollama
echo "Pulling Llama model in Ollama..."
docker exec ollama ollama pull wangshenzhi/llama3-8b-chinese-chat-ollama-q4

# Step 3: Build the Chatbot Docker Image
echo "Building chatbot Docker image..."
docker build -t chatbot-app .

# Step 4: Run the Chatbot Container
echo "Running chatbot container..."
# docker run -d --network chatbot-net --name chatbot-app -p 7860:7860 -e EMBED_DEVICE_CHOICE="cpu" chatbot-app
docker run -d --network chatbot-net --name chatbot-app -p 7860:7860 \
  -e EMBED_DEVICE_CHOICE="mps" \
  -e TOKENIZERS_PARALLELISM="false" chatbot-app

# Step 5: Display Logs
echo "Chatbot logs:"
docker logs -f chatbot-app

