FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt
RUN useradd --create-home demo
COPY --chown=demo:demo app.py serve.py ./
COPY --chown=demo:demo src ./src
COPY --chown=demo:demo templates ./templates
COPY --chown=demo:demo static ./static
COPY --chown=demo:demo model ./model
RUN chown demo:demo /app
USER demo
EXPOSE 8080
HEALTHCHECK --start-period=60s --interval=30s --timeout=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=8)"
CMD ["python", "serve.py"]
