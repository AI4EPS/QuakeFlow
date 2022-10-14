FROM continuumio/miniconda3

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Create the environment:
COPY env.yml /app
RUN conda env create --name quakeflow --file=env.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "quakeflow", "/bin/bash", "-c"]

# Copy files
COPY . /app

# Start API server
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "python", "producer.py"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "python", "producer_iris.py"]