FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 7860

# Run server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]