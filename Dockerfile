# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# (You'll need to create this file with your project dependencies)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# (if your application needs to expose any ports)
# EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]