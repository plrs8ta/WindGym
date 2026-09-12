"""python -m WindGym.plc_yaw {check,demo,replay,inspect}."""

import argparse
from dataclasses import replace

from .config import PLCYawConfig


def main():
    parser = argparse.ArgumentParser(
        description="PLC automatic-yaw execution reference"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Check Gymnasium and SB3 compatibility")
    check.add_argument("--config")
    demo_parser = sub.add_parser(
        "demo", help="Export eight zero-action execution scenarios"
    )
    demo_parser.add_argument("--config")
    demo_parser.add_argument("--output", required=True)
    replay_parser = sub.add_parser("replay", help="Replay averaged PLC inputs from CSV")
    replay_parser.add_argument("--config")
    replay_parser.add_argument("--input", required=True)
    replay_parser.add_argument("--output", required=True)
    inspect = sub.add_parser(
        "inspect", help="Inspect the read-only PLCopen reference; no code is executed"
    )
    inspect.add_argument("--xml", required=True)
    inspect.add_argument("--pou")
    args = parser.parse_args()
    if args.command == "inspect":
        from .reference import inspect_reference

        print(inspect_reference(args.xml, args.pou))
        return
    config = PLCYawConfig.load(args.config) if args.config else PLCYawConfig()
    if args.command == "check":
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check

        from .env import PLCYawEnv

        env = PLCYawEnv(config)
        try:
            check_env(env, skip_render_check=True)
            sb3_check(env)
            print(
                f"Gymnasium and SB3 checks passed: obs={env.observation_space.shape}, action=bounded vane correction"
            )
            print(
                "Python execution model checked; native PLC/runtime equivalence is not certified."
            )
        finally:
            env.close()
    elif args.command == "demo":
        from .replay import demo

        if not args.config:
            config = replace(config, episode_seconds=180)
        print(f"Eight execution timelines saved in {demo(args.output, config)}")
    else:
        from .replay import replay

        print(f"Input replay saved in {replay(args.input, args.output, config)}")


if __name__ == "__main__":
    main()
