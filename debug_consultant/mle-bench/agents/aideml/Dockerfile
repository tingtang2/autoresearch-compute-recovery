FROM mlebench-env

# Standard mle-bench directories
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

# Requirements for opencv and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the AIDE package root (parent build context: aide/ sits next to mle-bench/)
COPY aide ${AGENT_DIR}/aide

# Install AIDE ML from local source in editable mode
RUN conda run -n ${CONDA_ENV_NAME} bash -c "\
    pip install -e ${AGENT_DIR}/aide && \
    pip install --upgrade openai httpx" && \
    conda clean -afy

# Copy agent-specific files (use mle-bench/ prefix for parent context)
COPY mle-bench/agents/aideml ${AGENT_DIR}

# Make entire agent directory writable by nonroot user
# start.sh writes full_instructions.txt and AIDE creates logs/workspaces subdirs
RUN chmod -R a+rwx ${AGENT_DIR}
