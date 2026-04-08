FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose ports
EXPOSE 7860 7861

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run both backend and UI
CMD ["/app/start.sh"]