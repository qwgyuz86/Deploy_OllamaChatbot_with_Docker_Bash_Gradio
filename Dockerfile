# Define the base image
FROM python:3.9

# Set environment variables
ENV EMBED_DEVICE_CHOICE="cpu" \
    PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port for the Gradio app
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Assumes Ollama is running in a separate container and accessible at port 11434
# Start the app
# CMD python chatbot_app.py
CMD ["python", "chatbot_app.py"]