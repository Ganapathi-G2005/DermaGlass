
# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for CV libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# Note: We use the default pip install which will pick up the CPU version 
# because we will update requirements.txt to point to CPU wheels or just standard pypi 
# (standard pypi torch is huge, usually better to specify cpu index if needed, 
# but for simple deployment we'll rely on the updated requirements file)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security (required by some HF spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port (Hugging Face Spaces typically expects 7860)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
