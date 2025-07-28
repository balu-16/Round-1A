FROM --platform=linux/amd64 python:3.10-slim

# Create app directories
WORKDIR /app
RUN mkdir -p /app/input /app/output

# Install PyMuPDF (fitz), needed for PDF parsing
RUN pip install --no-cache-dir pymupdf

# Copy the parser script into the container
COPY main.py /app/main.py

# Set the locale to avoid Python UTF-8 issue
ENV PYTHONUTF8=1

# Set default environment variables for input and output (can be overridden)
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Entrypoint: run the script
ENTRYPOINT ["python3", "/app/main.py"]
