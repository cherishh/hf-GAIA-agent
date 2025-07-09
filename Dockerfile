# Use Python 3.12 base image with slim variant for smaller size
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    # Java for multi-language code execution
    openjdk-17-jdk \
    # OCR dependencies for pytesseract
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libtesseract-dev \
    # Image processing dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio processing dependencies
    ffmpeg \
    # Git for potential repository operations
    git \
    # Curl for downloading files
    curl \
    # SQLite for database operations
    sqlite3 \
    # Clean up apt cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for temporary files and data
RUN mkdir -p /tmp/agent_files /app/temp_data

# Set proper permissions
RUN chmod +x /app && \
    chmod 777 /tmp/agent_files /app/temp_data

# Expose port for Jupyter notebook (optional, since you're using test.ipynb)
EXPOSE 8888

# Expose port for Gradio app (if needed)
EXPOSE 7860

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/agent_files
USER appuser

# Default command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]