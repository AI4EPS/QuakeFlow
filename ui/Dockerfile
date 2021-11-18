FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY env.yml /app
RUN conda env create --name quakeflow --file=env.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "quakeflow", "/bin/bash", "-c"]

# Copy files
COPY . /app

# Expose API port
EXPOSE 8005

ENV PYTHONUNBUFFERED=1

# Start API server
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "python", "app_plotly.py"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "gunicorn", "app_plotly:server", "--reload", "-b", "0.0.0.0:8005"]
