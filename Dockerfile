# Reproducible environment for LpH-CEG artifact evaluation.
#
#   docker build -t lph-ceg .
#   docker run --rm lph-ceg                      # tests + lint + figures
#   docker run --rm lph-ceg bash scripts/reproduce_all.sh experiments
#
# Real-data experiments need the datasets placed under data/ (see data/README.md);
# synthetic experiments, unit tests, lint, and figures run with no external data.
FROM python:3.10.12-slim

WORKDIR /work

# Pinned scientific stack (reproduces reported numbers).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
ENV PYTHONPATH=/work
ENV PYTHONDONTWRITEBYTECODE=1

# Default: quick reproduction (tests + lint + figures). The paper build
# (scripts/reproduce_all.sh paper) additionally requires a LaTeX toolchain
# (e.g. apt-get install texlive-latex-extra), omitted here to keep the image small.
CMD ["bash", "scripts/reproduce_all.sh", "quick"]
