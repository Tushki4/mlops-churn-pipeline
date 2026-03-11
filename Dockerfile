# Base image — slim Python 3.11 on Debian
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# ── LAYER CACHING TRICK ──────────────────────────────────────
# Copy requirements FIRST and install dependencies.
# This layer only rebuilds when requirements.txt changes.
# If you copied all code first, every code change would
# reinstall all dependencies — very slow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY src/ ./src/
COPY data/ ./data/

# Expose the port the API runs on
EXPOSE 8000

# Start the API when container runs
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]