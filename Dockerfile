FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev

# Set working directory to the correct path
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . . 

# Change the working directory to app/
WORKDIR /app/app

# Expose necessary ports
EXPOSE 5000

# Command to run the Flask server
CMD ["python", "server.py"]
