# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for scipy and audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && apt-get clean

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install kokoro --no-dependencies

# Copy the application files
COPY . .

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
