# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS base

LABEL org.opencontainers.image.title="Lattice-QFT-Quantum-Simulation"
LABEL org.opencontainers.image.description="Quantum simulation framework for lattice quantum field theories"
LABEL org.opencontainers.image.source="https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation"
LABEL org.opencontainers.image.licenses="MIT"

# Python runtime settings
# Add mpl backend to avoid issues with plt.show
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Create a non-root user
RUN groupadd --system quantum \
    && useradd \
        --system \
        --gid quantum \
        --create-home \
        --home-dir /home/quantum \
        quantum

# Copy dependency metadata first to improve Docker layer caching.
COPY requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

# Copy the source code after dependencies.
COPY QuantumSimulation/ ./QuantumSimulation/
COPY configs/ ./configs/
COPY results/ ./results/
COPY README.md LICENSE CITATION.cff ./

# Install the local package without reinstalling dependencies.
RUN python -m pip install --no-deps .

# Ensure generated files can be written by the non-root user.
RUN chown -R quantum:quantum /app

USER quantum

# Default command: import smoke test.
CMD ["python", "-c", "import QuantumSimulation; print('QuantumSimulation container is ready.')"]
