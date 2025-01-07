# Deploy Ollama Chatbot with Docker Bash Gradio

### Introduction
Ollama is an open-source tool that allows users to run, create, and share large language models (LLMs) locally on their own computer.
It is particularly appealing to businesses concerned with data control and privacy. It allows users to maintain full data ownership and avoid the potential security risks.

To launch a chatbot app built using Ollama, it involves these major steps:

1. Download Ollama
2. Pull the Llama model you want to use
3. Start Ollama server to serve the pulled model
4. Run the app

To package the ollama setup, app codes and dependencies as Docker containers for deployment, this repository provides the codes with two methods:
1. Docker + Bash
2. Docker-compose with a yml file

The challenge to make Ollama to work in docker is that you need to make sure the dependencies between the containers and their network are working in the desired order.
Please see below for deployment instruction and the files in the repo for details.

### Deployment Method 1: Docker + Bash

In the terminal, CD to the directory where the folder with the repo files are located, run this command:
`./deploy_chatbot.sh`

If you use method 1, you do not need the `docker-compose.yml` file in the folder

### Deployment Method 2: Docker-compose

In the terminal, CD to the directory where the folder with the repo files are located, run this command:
`docker-compose up --build`

If you use method 2, you do not need the bash script `deploy_chatbot.sh` in the folder.
