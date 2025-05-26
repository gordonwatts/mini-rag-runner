# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package and dependencies
RUN pip install --upgrade pip && \
    pip install .

# Expose port (adjust if your webapi uses a different port)
EXPOSE 8001

# Set the entrypoint to allow passing arguments to the command
ENTRYPOINT ["light-rag-webapi"]
CMD []
