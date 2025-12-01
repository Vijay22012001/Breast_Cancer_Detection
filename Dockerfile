# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies needed for waitress (essential for building packages like scikit-learn)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL CORRECTION: COPY entire directories ---

# Copy the Flask application code
COPY application.py .

# Copy the directory containing the models
COPY models/ models/ 

# Copy the directory containing the HTML templates
COPY templates/ templates/

# --- END CRITICAL CORRECTION ---

# Expose the port the app will run on 
EXPOSE 5000

# Command to run the application using Waitress
# This assumes your Flask app instance is named 'app' in the 'app.py' file.
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "application:app"]