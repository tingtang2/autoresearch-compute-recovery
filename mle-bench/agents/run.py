import logging
import shutil
import time
from pathlib import Path

import docker
from docker.models.containers import Container
from dotenv import dotenv_values

from agents.registry import Agent
from environment.utils import (
    create_competition_container,
    extract_from_container,
    extract_from_container_sysbox,
)
from mlebench.registry import Competition
from mlebench.utils import purple

CONSTANTS = dotenv_values(Path(__file__).parent.resolve() / ".shared_env")


def save_output(container: Container, save_dir: Path, container_config: dict, agent: Agent = None) -> Path:
    """
    Extracts the submission, logs, and code directories from the container

    and saves them to the specified directory.

    Args:
        container: The Docker container.
        save_dir: The directory where the output file will be saved.
        container_config: The container configuration.
        agent: Optional agent object to handle agent-specific extraction logic.
    Returns:
        Path to the output directory.
    """
    if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
        extraction_fn = extract_from_container_sysbox
    else:
        extraction_fn = extract_from_container

    for dir_type in ["SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"]:
        container_dir = CONSTANTS[dir_type]
        extraction_fn(container, container_dir, save_dir)

    # Agent-specific extraction logic for agents that save files to non-standard locations
    if agent and agent.id in ["mlmaster", "opendevin", "rdagent", "aide"]:
        logger = logging.getLogger(__name__)
        agent_dir = CONSTANTS["AGENT_DIR"]
        
        # Check and extract submission file if missing
        submission_path = save_dir / "submission" / "submission.csv"
        if not submission_path.exists():
            agent_name = agent.id
            logger.info(f"{agent_name}: No submission.csv found in SUBMISSION_DIR, searching workspace directories...")
            try:
                # Different search patterns for different agents
                if agent.id == "mlmaster":
                    find_cmd = f"find {agent_dir}/workspaces -type f \\( -path '*/best_submission/submission.csv' -o -path '*/submission/submission.csv' \\) 2>/dev/null | sort -r | head -1"
                elif agent.id == "aide":
                    # Aide writes to workspaces/exp/submission/submission.csv or workspaces/exp/best_submission/submission.csv
                    find_cmd = f"find {agent_dir}/workspaces -type f \\( -path '*/best_submission/submission.csv' -o -path '*/submission/submission.csv' \\) 2>/dev/null | sort -r | head -1"
                elif agent.id == "opendevin":
                    # OpenDevin might save submissions in CODE_DIR or workspace directories
                    # Search broadly for any submission.csv files
                    find_cmd = f"find {CONSTANTS['CODE_DIR']} {agent_dir} -type f -name 'submission.csv' 2>/dev/null | sort -r | head -1"
                else:
                    find_cmd = f"find {agent_dir} -type f -name 'submission.csv' 2>/dev/null | sort -r | head -1"
                
                exit_code, output = container.exec_run(
                    f"sh -c '{find_cmd}'",
                    user="nonroot"
                )
                
                if exit_code == 0 and output:
                    workspace_path = output.decode("utf-8").strip()
                    if workspace_path:
                        logger.info(f"{agent_name}: Found submission at {workspace_path}, copying to submission directory...")
                        copy_cmd = f"mkdir -p {CONSTANTS['SUBMISSION_DIR']} && cp '{workspace_path}' {CONSTANTS['SUBMISSION_DIR']}/submission.csv"
                        copy_exit, _ = container.exec_run(
                            f"sh -c '{copy_cmd}'",
                            user="nonroot"
                        )
                        if copy_exit == 0:
                            logger.info(f"{agent_name}: Successfully copied submission file")
                            # Re-extract submission directory
                            if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
                                extract_from_container_sysbox(container, CONSTANTS["SUBMISSION_DIR"], save_dir)
                            else:
                                extract_from_container(container, CONSTANTS["SUBMISSION_DIR"], save_dir)
                        else:
                            logger.warning(f"{agent_name}: Failed to copy submission file, exit code: {copy_exit}")
                    else:
                        logger.info(f"{agent_name}: No submission file found in workspace directories")
                else:
                    logger.info(f"{agent_name}: Find command for submission returned exit code {exit_code}")
            except Exception as e:
                logger.warning(f"{agent_name}: Error searching for submission file in workspace: {e}")
        
        # Check and extract code files if missing (ML-Master specific)
        if agent.id == "mlmaster":
            code_path = save_dir / "code"
            try:
                code_dir_empty = not code_path.exists() or not any(code_path.iterdir())
            except Exception:
                code_dir_empty = True
            if code_dir_empty:
                logger.info("ML-Master: No code found in CODE_DIR, searching workspace directories...")
                try:
                    # Find best_solution directory containing solution.py
                    find_cmd = f"find {agent_dir}/workspaces -type f -path '*/best_solution/solution.py' 2>/dev/null | sort -r | head -1"
                    exit_code, output = container.exec_run(
                        f"sh -c '{find_cmd}'",
                        user="nonroot"
                    )
                    
                    if exit_code == 0 and output:
                        solution_py_path = output.decode("utf-8").strip()
                        if solution_py_path:
                            best_solution_dir = solution_py_path.replace("/solution.py", "")
                            logger.info(f"ML-Master: Found best_solution directory at {best_solution_dir}, copying to code directory...")
                            # Copy all files from best_solution directory to CODE_DIR
                            # This should include solution.py and node_id.txt
                            copy_cmd = f"mkdir -p {CONSTANTS['CODE_DIR']} && (find '{best_solution_dir}' -maxdepth 1 -type f -exec cp {{}} {CONSTANTS['CODE_DIR']}/ \\;)"
                            copy_exit, _ = container.exec_run(
                                f"sh -c '{copy_cmd}'",
                                user="nonroot"
                            )
                            if copy_exit == 0:
                                logger.info("ML-Master: Successfully copied code files")
                                # Re-extract code directory
                                if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
                                    extract_from_container_sysbox(container, CONSTANTS["CODE_DIR"], save_dir)
                                else:
                                    extract_from_container(container, CONSTANTS["CODE_DIR"], save_dir)
                            else:
                                logger.warning(f"ML-Master: Failed to copy code files, exit code: {copy_exit}")
                        else:
                            logger.info("ML-Master: No solution.py found in workspace directories")
                    else:
                        logger.info(f"ML-Master: Find command for code returned exit code {exit_code}")
                except Exception as e:
                    logger.warning(f"ML-Master: Error searching for code files in workspace: {e}")

        # Check and extract code files if missing (RD-Agent)
        if agent.id == "rdagent":
            code_path = save_dir / "code"
            try:
                code_dir_empty = not code_path.exists() or not any(code_path.iterdir())
            except Exception:
                code_dir_empty = True
            if code_dir_empty:
                logger.info("RD-Agent: No code found in CODE_DIR, searching agent runtime workspace...")
                try:
                    # Prefer a workspace containing main.py; fall back to any .py.
                    find_main_cmd = (
                        f"find {agent_dir}/runtime/workspace -type f -name 'main.py' 2>/dev/null | "
                        "sort -r | head -1"
                    )
                    exit_code, output = container.exec_run(f"sh -c '{find_main_cmd}'", user="nonroot")
                    candidate = output.decode("utf-8", errors="replace").strip() if output else ""

                    if exit_code != 0 or not candidate:
                        find_any_cmd = (
                            f"find {agent_dir}/runtime/workspace -type f -name '*.py' 2>/dev/null | "
                            "sort -r | head -1"
                        )
                        exit_code, output = container.exec_run(f"sh -c '{find_any_cmd}'", user="nonroot")
                        candidate = output.decode("utf-8", errors="replace").strip() if output else ""

                    if candidate:
                        ws_dir = candidate.rsplit("/", 1)[0]
                        logger.info(f"RD-Agent: Found workspace at {ws_dir}, copying into CODE_DIR...")
                        # Copy the whole workspace folder (robust across images where `cp --parents` may be missing).
                        copy_cmd = (
                            f"mkdir -p {CONSTANTS['CODE_DIR']} && "
                            f"rm -rf {CONSTANTS['CODE_DIR']}/workspace && "
                            f"cp -r '{ws_dir}' {CONSTANTS['CODE_DIR']}/workspace"
                        )
                        copy_exit, _ = container.exec_run(f"sh -c '{copy_cmd}'", user="nonroot")
                        if copy_exit == 0:
                            logger.info("RD-Agent: Successfully copied workspace code into CODE_DIR")
                            # Re-extract code directory
                            if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
                                extract_from_container_sysbox(container, CONSTANTS["CODE_DIR"], save_dir)
                            else:
                                extract_from_container(container, CONSTANTS["CODE_DIR"], save_dir)
                        else:
                            logger.warning(f"RD-Agent: Failed to copy code files, exit code: {copy_exit}")
                    else:
                        logger.info("RD-Agent: No workspace code found under /home/agent/runtime/workspace")
                except Exception as e:
                    logger.warning(f"RD-Agent: Error searching for code files in workspace: {e}")

    return save_dir


def execute_agent(container: Container, agent: Agent, logger: logging.Logger):
    """
    Initiates the agent via its start script inside the container.
    """
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    logger.info("Running agent...")
    exit_code, output = container.exec_run(cmd, stream=True, user="nonroot")

    for chunk in output:
        # Be robust to split/malformed UTF-8 sequences in streamed chunks
        try:
            line = chunk.decode("utf-8", errors="replace").strip()
        except Exception:
            # Last-resort safeguard; represent as repr of bytes
            line = repr(chunk)
        logger.info(f"[Container] {line}")


def clean_up(container: Container, logger: logging.Logger, retain: bool = False) -> bool:
    """
    Stops and removes the container.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Cleaning up container: {container.name}")
    try:
        container.stop()
        if not retain:
            container.remove()
        logger.info(f"Container {container.name} stopped and removed.")
        return True
    except Exception as e:
        logger.error(
            f"Error cleaning up: {e}. You may wish to manually check the status of the {container.name} container."
        )
        return False


def run_in_container(
    client: docker.DockerClient,
    competition: Competition,
    agent: Agent,
    image: str,
    container_config: dict,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
) -> Path:
    """
    Runs environment containing the competition and agent for a set maximum amount of time.

    Args:
        client: Docker client.
        competition: The competition to run.
        agent: The agent to run.
        image: The Docker image to use. Assumes the image is built.
        container_config: Configuration for the Docker container.
        retain_container: Whether to retain the container after the run instead of removing it.
        run_dir: Path to the directory where all assets associated with the run are stored.
        logger: Logger for the run.

    Returns:
        Path to the output file.
    """
    volumes_config = {
        competition.public_dir.resolve().as_posix(): {
            "bind": "/home/data",
            "mode": "ro",
        },
        competition.private_dir.resolve().as_posix(): {
            "bind": f"/private/data/{competition.id}/prepared/private/",
            "mode": "ro",
        },
    }

    # Bind-mount submission/logs/code dirs for real-time access on the host
    for dir_type, container_path in [
        ("submission", CONSTANTS["SUBMISSION_DIR"]),
        ("logs", CONSTANTS["LOGS_DIR"]),
        ("code", CONSTANTS["CODE_DIR"]),
    ]:
        host_dir = run_dir / dir_type
        host_dir.mkdir(parents=True, exist_ok=True)
        volumes_config[host_dir.resolve().as_posix()] = {
            "bind": container_path,
            "mode": "rw",
        }

    container = create_competition_container(
        client=client,
        competition=competition,
        container_config=container_config,
        volumes_config=volumes_config,
        env_vars={
            "COMPETITION_ID": competition.id,
            **agent.env_vars,
        },
        container_image=image,
        privileged=agent.privileged,
    )

    logger.info(purple(f"Run started: {run_dir}"))
    try:
        time_start = time.monotonic()
        container.start()
        exit_code, _ = container.exec_run(
            'timeout 60s sh -c "while ! curl -s http://localhost:5000/health > /dev/null; do sleep 1; done"'
        )
        if exit_code != 0:
            raise RuntimeError(
                "The grading server failed to start within 60 seconds. This is likely due to an error in `entrypoint.sh`; check the logs."
            )
        execute_agent(container, agent, logger)
        save_output(container, run_dir, container_config, agent)
        time_end = time.monotonic()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")
        return run_dir
    except Exception as e:
        raise e
    finally:
        clean_up(container, logger, retain_container)
