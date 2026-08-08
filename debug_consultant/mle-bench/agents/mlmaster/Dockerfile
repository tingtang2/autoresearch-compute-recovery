FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
# (parent build context: ML-Master source lives in mlmaster/ next to mle-bench/)
COPY mlmaster/requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv and other system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} bash -c "pip install -r ${AGENT_DIR}/requirements.txt" && \
    conda clean -afy

# put all the agent files in the expected location:
#   ML-Master source, then the mle-bench wrapper (start.sh, config.yaml, notes)
COPY mlmaster ${AGENT_DIR}
COPY mle-bench/agents/mlmaster ${AGENT_DIR}
RUN chmod -R a+rw ${AGENT_DIR}

# Make s5e12 grade.py writable so start.sh can patch it at runtime
RUN chmod a+w /mlebench/mlebench/competitions/playground-series-s5e12/grade.py 2>/dev/null || true
