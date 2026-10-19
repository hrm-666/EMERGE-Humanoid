"""Start the existing room, SONIC and learned policies as the desktop user.

Simulation transport is fixed to loopback; no privileged namespaces or sudo.
The Controller starts the sole Executive. This launcher never starts it twice.
"""
from __future__ import annotations

import argparse
import os
import shlex
import socket
from pathlib import Path

from robot.drivers.official01_executive import ExecutiveProcessClient

ROOT = Path(__file__).resolve().parents[2]


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Return the selected simulation service and its private environment.

    Args:
        args: Parsed CLI settings, including service and output directory.

    Returns:
        An argument list and environment for a user-owned process.

    Raises:
        ValueError: A graphical simulator has no desktop display.
    """
    home = Path.home()
    bridge = Path(
        os.environ.get("VISIONPRO_SONIC_ROOT",
                       home / "visionpro_sonic_bridge")).expanduser()
    core = Path(os.environ.get("GR00T_ROOT",
                               home / "GR00T-WholeBodyControl")).expanduser()
    isaac = Path(os.environ.get("ISAAC_ROOT",
                                home / "Isaac-GR00T-N1.7")).expanduser()
    lightnav = Path(os.environ.get("LIGHTNAV_ROOT",
                                   home / "LightNav-0")).expanduser()
    env = ExecutiveProcessClient(bridge)._runtime_env()
    output = Path(args.output_dir).expanduser().resolve()
    if args.service == "sim":
        if not args.headless and not env.get("DISPLAY"):
            raise ValueError(
                "Start the viewer on the lab desktop or select --headless.")
        command = [
            str(core / ".venv_sim/bin/python"),
            str(ROOT / "scripts/G1_eval/sim_runtime.py"),
            "--model",
            str(bridge /
                "integrations/inspire_mujoco/scenes/official01_room.xml"),
            "--interface",
            "lo",
            "--domain",
            "0",
            "--observation-endpoint",
            "tcp://127.0.0.1:5560",
            "--observation-mirror-endpoint",
            "ipc:///tmp/official01_emerge_observation.ipc",
            "--publish-third-person",
            "--observation-hz",
            "10",
            "--duration",
            str(args.duration),
            "--output-dir",
            str(output),
            "--screenshot",
            str(output / "final_third_person.png"),
            "--ego-screenshot",
            str(output / "final_ego.png"),
            "--metrics",
            str(output / "task_metrics.json"),
        ]
        if args.headless:
            command.append("--headless")
            env["MUJOCO_GL"] = "egl"
    elif args.service == "sonic":
        deploy = core / "gear_sonic_deploy"
        command = [
            "/usr/bin/bash",
            str(bridge /
                "integrations/sonic_sim_isolation/launch_sonic_deploy_with_ros_env.sh"
               ),
            str(deploy / "target/release/g1_deploy_onnx_ref"),
            "lo",
            str(deploy / "policy/release/model_decoder.onnx"),
            str(deploy / "reference/example"),
            "--obs-config",
            str(deploy / "policy/release/observation_config_zmq_ardy_g1.yaml"),
            "--encoder-file",
            str(deploy / "policy/release/model_encoder.onnx"),
            "--planner-file",
            str(deploy / "planner/target_vel/V2/planner_sonic.onnx"),
            "--input-type",
            "zmq_manager",
            "--output-type",
            "zmq",
            "--zmq-host",
            "127.0.0.1",
            "--zmq-port",
            "5556",
            "--zmq-out-port",
            "5557",
            "--disable-crc-check",
            "--disable-dex3-hands",
            "--logs-dir",
            str(output),
            "--enable-csv-logs",
        ]
    elif args.service == "gr00t":
        checkpoint = Path(args.checkpoint).expanduser()
        command = [
            str(isaac / ".venv/bin/python"),
            str(isaac / "gr00t/eval/run_gr00t_server.py"), "--model-path",
            str(checkpoint), "--embodiment-tag", "NEW_EMBODIMENT", "--device",
            "cuda:0", "--host", "127.0.0.1", "--port", "5550"
        ]
    else:
        env.update(HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1",
                   CUDA_VISIBLE_DEVICES="0",
                   VLN_KV_CACHE_GIB="1",
                   VLN_VLLM_ENFORCE_EAGER="1")
        env["PYTHONPATH"] = str(lightnav / "src")
        command = [
            str(home / "miniconda3/envs/lightnav-0/bin/python"), "-m",
            "lightnav.serving.ws_server", "--task", "vln", "--model_path",
            str(lightnav / "checkpoints/LightNav-0"), "--backend", "vllm_local",
            "--host", "127.0.0.1", "--port", "8051", "--max_batch_size", "1",
            "--gpu_memory_utilization", "0.50", "--record_dir",
            str(output)
        ]
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service",
                        choices=("sim", "sonic", "gr00t", "lightnav"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--output-dir",
                        default=str(ROOT / "outputs/g1_integration"))
    parser.add_argument(
        "--checkpoint",
        default=str(
            Path.home() /
            "Isaac-GR00T-N1.7/checkpoints/official03pick80_official02pick20_from_ckpt20000_v1/checkpoint-15000"
        ))
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    command, env = build_command(args)
    print("SIMULATION_ONLY: user-owned processes; DDS interface=lo", flush=True)
    print(shlex.join(command), flush=True)
    if args.service == "sonic":
        print("After Init Done, press ] once to start standing and feedback.",
              flush=True)
    if args.print_command:
        return
    port = {
        "sim": 5560,
        "sonic": 5557,
        "gr00t": 5550,
        "lightnav": 8051
    }[args.service]
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"Port {port} is already in use; inspect the existing service first."
            )
    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
