FROM mlebench-env

# MLE-bench standard environment variables
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

# Create standard directories
RUN mkdir -p ${SUBMISSION_DIR} ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

# RD-Agent's LocalEnv (conda mode) creates symlinks to emulate Docker volumes.
# The MLEBench config maps data to /mle/data. Pre-create with nonroot access.
RUN mkdir -p /mle && chmod 777 /mle

# Copy official microsoft/RD-Agent into the container
COPY RD-Agent-official /opt/rdagent

# Install RD-Agent as editable in the agent conda env
ARG CONDA_ENV_NAME=agent
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
COPY mle-bench/agents/rdagent/requirements.txt /tmp/rdagent-requirements.txt
# strace + coverage are required by RD-Agent's check_runtime_environment()
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*
RUN conda run -n ${CONDA_ENV_NAME} pip install -e /opt/rdagent coverage && \
    conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/rdagent-requirements.txt && \
    conda clean -afy

# RD-Agent's get_ds_env() looks up conda envs named "kaggle" or "mlebench".
# Symlink them to the actual "agent" env so no monkey-patching is needed.
RUN ln -s /opt/conda/envs/${CONDA_ENV_NAME} /opt/conda/envs/kaggle && \
    ln -s /opt/conda/envs/${CONDA_ENV_NAME} /opt/conda/envs/mlebench

# Copy agent files (start.sh, etc.)
COPY mle-bench/agents/rdagent/ ${AGENT_DIR}/
RUN chmod +x ${AGENT_DIR}/start.sh && \
    chown -R nonroot:nonroot ${AGENT_DIR}

WORKDIR ${AGENT_DIR}

CMD ["/bin/bash"]
