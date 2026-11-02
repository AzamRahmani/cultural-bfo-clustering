# Reproducible environment for Project A.
#
# The image is built from a committed Git state, never from the working tree,
# so uncommitted edits cannot leak into a reproduction. Build a frozen state
# with --build-arg GIT_REF=project-a-v1.
FROM python:3.14.5-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY .git /tmp/source/.git
ARG GIT_REF=HEAD
RUN git clone --quiet --no-checkout /tmp/source /app \
    && git -C /app checkout --quiet --detach "$GIT_REF" \
    && rm -rf /tmp/source

WORKDIR /app
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Cache the OpenML datasets so tests and analysis can run without network.
RUN python -c "from data_loader import load_standardized_dataset as load; [load(name) for name in ('iris', 'wine', 'glass', 'cancer')]"

CMD ["python", "test_project.py"]
