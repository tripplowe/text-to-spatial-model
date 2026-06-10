FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and prompt assets
COPY app.py .
COPY system_prompt.txt .
COPY gis_ontology.json .

# Logs are written here; mounted as a volume in compose so they persist
RUN mkdir -p /app/logs

EXPOSE 8080

# Bind to 0.0.0.0 INSIDE the container; the host port mapping in
# docker-compose.yml restricts external exposure to loopback (127.0.0.1).
ENV BIND_HOST=0.0.0.0
ENV BIND_PORT=8080

CMD ["python", "app.py"]
