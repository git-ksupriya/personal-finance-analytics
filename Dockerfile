# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PostgreSQL and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY Personal_Finance/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    psycopg2-binary \
    passlib \
    bcrypt \
    itsdangerous \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    joblib \
    pandas \
    numpy \
    python-multipart

# Copy project files
COPY Personal_Finance /app/Personal_Finance

# Expose port (8000 for local / Azure App Service)
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "Personal_Finance.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
