"""Train with curriculum learning - automatically chain training stages and load
model parameters from previous stages. Supports agent count scaling (e.g. manyagent_swimmer
2x3 -> 4x3 -> 6x3).
"""

import argparse
import copy
import json
import os
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mamujoco",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="curriculum_training",
        help="Experiment name.",
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to env config yaml. Overrides --env for loading env config.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except Exception:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)

    if args["load_config"] != "":
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # Override env config from --config if provided
    if args.get("config"):
        import yaml
        with open(args["config"], "r", encoding="utf-8") as f:
            env_override = yaml.load(f, Loader=yaml.FullLoader)
        env_args.update(env_override)

    update_args(unparsed_dict, algo_args, env_args)

    if args["env"] == "dexhands":
        import isaacgym  # noqa: F401

    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # Check curriculum learning config
    curriculum_cfg = env_args.get("curriculum_learning", {})
    if not curriculum_cfg.get("enabled", False):
        print("Curriculum learning is disabled. Enable it in env config:")
        print("  curriculum_learning:")
        print("    enabled: true")
        print("    stages:")
        print('      - agent_conf: "2x3"')
        print("        num_env_steps: 5000000")
        print('      - agent_conf: "4x3"')
        print("        num_env_steps: 5000000")
        print("        load_from_previous: true")
        return

    stages = curriculum_cfg.get("stages", [])
    if not stages:
        print("No curriculum stages defined. Add stages under curriculum_learning.stages")
        return

    from harl.runners import RUNNER_REGISTRY

    prev_models_dir = None
    prev_num_agents = None

    for stage_idx, stage in enumerate(stages):
        agent_conf = stage.get("agent_conf")
        num_env_steps = stage.get("num_env_steps", algo_args["train"]["num_env_steps"])
        load_from_previous = stage.get("load_from_previous", False)

        if agent_conf is None:
            print(f"Stage {stage_idx + 1}: agent_conf not set, skipping")
            continue

        # Update env_args for this stage
        stage_env_args = copy.deepcopy(env_args)
        stage_env_args["agent_conf"] = agent_conf

        # Update algo_args (deep copy to avoid mutating shared config)
        stage_algo_args = copy.deepcopy(algo_args)
        stage_algo_args["train"]["num_env_steps"] = num_env_steps

        if load_from_previous and prev_models_dir is not None:
            stage_algo_args["train"]["model_dir"] = prev_models_dir
            stage_algo_args["train"]["curriculum_restore"] = True
            stage_algo_args["train"]["curriculum_source_num_agents"] = prev_num_agents
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}/{len(stages)}: agent_conf={agent_conf}")
            print(f"  Loading from previous stage: {prev_models_dir}")
            print(f"  (curriculum transfer with partial loading)")
        else:
            stage_algo_args["train"]["model_dir"] = None
            stage_algo_args["train"]["curriculum_restore"] = False
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}/{len(stages)}: agent_conf={agent_conf} (from scratch)")

        runner = RUNNER_REGISTRY[args["algo"]](args, stage_algo_args, stage_env_args)
        runner.run()
        prev_models_dir = runner.save_dir
        prev_num_agents = runner.num_agents
        runner.close()

        print(f"Stage {stage_idx + 1} done. Models saved to {prev_models_dir}")

    print("\n" + "=" * 60)
    print("Curriculum training completed.")


if __name__ == "__main__":
    main()
