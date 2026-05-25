"""Train an algorithm.

Supports optional curriculum learning (CL) and optional LLM-guided auto adjustment
for on-policy runners. Enable them via `--curriculum True/False` and `--llm True/False`.
"""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v!r}")


def init_llm_client(env_args=None):
    """Initialize an OpenAI-compatible client (optional dependency).

    Uses environment variables:
      - LLM_API_KEY
      - LLM_BASE_URL
      - LLM_MODEL

    Optionally, if env vars are not set, reads from env yaml:
      env_args["llm"]["api_key" | "base_url" | "model"]
    (Not recommended to commit secrets into the repo; prefer a gitignored override file.)
    """
    import os

    llm_cfg = (env_args or {}).get("llm", {}) if isinstance(env_args, dict) else {}
    api_key = os.getenv("LLM_API_KEY") or llm_cfg.get("api_key")
    base_url = os.getenv("LLM_BASE_URL") or llm_cfg.get("base_url")
    model = os.getenv("LLM_MODEL") or llm_cfg.get("model") or "qwen3.5-plus"

    if not api_key:
        print("LLM disabled: missing `LLM_API_KEY` (env var) and `llm.api_key` (env yaml).")
        return None, None

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(f"LLM disabled: cannot import `openai` package ({e}).")
        return None, None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return client, model
    except Exception as e:
        print(f"LLM disabled: client init failed ({e}).")
        return None, None


def _supports_llm_on_policy(runner) -> bool:
    required = ("warmup", "collect", "insert", "compute", "train", "after_update", "prep_rollout", "prep_training")
    return all(hasattr(runner, name) for name in required)


def _get_llm_guidance(client, model, training_context, current_params):
    if client is None:
        return "LLM client unavailable."
    prompt = f"""
你在 MaMuJoCo 环境中训练 MAPPO（PPO 类 on-policy）。你的任务是基于训练日志与当前超参，给出“下一阶段”可执行的超参微调建议。

【输入】
ENV:
{training_context.get('env_meta')}

METRICS:
{training_context.get('metrics')}

CURRENT_PARAMS:
{current_params}

CONSTRAINTS:
- 只允许建议调整以下参数（不在列表的一律不要输出）：lr, critic_lr, entropy_coef, value_loss_coef, std_x_coef, std_y_coef, gamma, gae_lambda
- 每次最多调整 3 个参数；每个参数的新值必须是数值（float）
- 建议要保守、渐进：除非有明显证据，不要大幅跳变（例如 lr/critic_lr 单次变化不超过 2x）
- 安全边界（建议必须落在范围内）：
{training_context.get('safety_bounds')}

【输出】
只输出一个 JSON 对象（不要输出任何额外文本、不要 markdown 代码块），必须符合下面 schema：
{{
  "analysis": "一句话概括当前训练问题/阶段",
  "adjustments": {{
    "lr": 0.0003,
    "entropy_coef": 0.005
  }},
  "rationale": ["原因1", "原因2"],
  "confidence": 0.0
}}

如果不建议调整任何参数：adjustments 设为 {{}}。
confidence 取值范围 [0,1]。
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是 MAPPO 训练调参助手。严格只输出 JSON（无多余文本），并确保输出可被 json.loads 解析。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM call failed: {e}"


def _format_training_context(episode, total_episodes, actor_train_infos, critic_train_info):
    context = f"当前训练进度: 第{episode}/{total_episodes}个训练周期\n\nActor训练信息:\n"
    for i, info in enumerate(actor_train_infos):
        context += f"Agent {i}: "
        if isinstance(info, dict):
            for key, value in info.items():
                try:
                    context += f"{key}: {float(value):.4f}, "
                except Exception:
                    context += f"{key}: {value}, "
        context += "\n"
    context += "\nCritic训练信息:\n"    
    if isinstance(critic_train_info, dict):
        for key, value in critic_train_info.items():
            try:
                context += f"{key}: {float(value):.4f}\n"
            except Exception:
                context += f"{key}: {value}\n"
    return context


def _format_current_params(algo_args):
    model = algo_args.get("model", {})
    algo = algo_args.get("algo", {})
    return (
        f"lr: {model.get('lr')}\n"
        f"critic_lr: {model.get('critic_lr')}\n"
        f"entropy_coef: {algo.get('entropy_coef')}\n"
        f"value_loss_coef: {algo.get('value_loss_coef')}\n"
        f"std_x_coef: {model.get('std_x_coef')}\n"
        f"std_y_coef: {model.get('std_y_coef')}\n"
        f"gamma: {algo.get('gamma')}\n"
        f"gae_lambda: {algo.get('gae_lambda')}\n"
    )


def _parse_llm_suggestions(text):
    import re

    suggestions = {}
    param_patterns = {
        "lr": [r"lr[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"lr[:：]\s*([\d.]+)"],
        "critic_lr": [r"critic_lr[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"critic_lr[:：]\s*([\d.]+)"],
        "entropy_coef": [r"entropy_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"entropy_coef[:：]\s*([\d.]+)"],
        "value_loss_coef": [
            r"value_loss_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)",
            r"value_loss_coef[:：]\s*([\d.]+)",
        ],
        "std_x_coef": [r"std_x_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"std_x_coef[:：]\s*([\d.]+)"],
        "std_y_coef": [r"std_y_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"std_y_coef[:：]\s*([\d.]+)"],
        "gamma": [r"gamma[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"gamma[:：]\s*([\d.]+)"],
        "gae_lambda": [r"gae_lambda[:：]\s*[\d.]+\s*->\s*([\d.]+)", r"gae_lambda[:：]\s*([\d.]+)"],
    }
    for name, patterns in param_patterns.items():
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                suggestions[name] = float(m.group(1))
                break
            except Exception:
                continue
    return suggestions


def _try_parse_llm_json(text):
    import json
    import re

    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _extract_llm_adjustments(llm_output):
    """Extract parameter adjustments from LLM output.

    Prefer strict JSON per prompt; fall back to regex parsing.
    """
    obj = _try_parse_llm_json(llm_output)
    if isinstance(obj, dict) and isinstance(obj.get("adjustments"), dict):
        adjustments = {}
        for k, v in obj["adjustments"].items():
            try:
                adjustments[str(k)] = float(v)
            except Exception:
                continue
        return adjustments
    return _parse_llm_suggestions(llm_output)


def _get_param_value(algo_args, name):
    mapping = {
        "lr": ("model", "lr"),
        "critic_lr": ("model", "critic_lr"),
        "entropy_coef": ("algo", "entropy_coef"),
        "value_loss_coef": ("algo", "value_loss_coef"),
        "std_x_coef": ("model", "std_x_coef"),
        "std_y_coef": ("model", "std_y_coef"),
        "gamma": ("algo", "gamma"),
        "gae_lambda": ("algo", "gae_lambda"),
    }
    if name not in mapping:
        return None
    section, key = mapping[name]
    return algo_args.get(section, {}).get(key)


def _set_param_value(algo_args, name, value) -> bool:
    mapping = {
        "lr": ("model", "lr"),
        "critic_lr": ("model", "critic_lr"),
        "entropy_coef": ("algo", "entropy_coef"),
        "value_loss_coef": ("algo", "value_loss_coef"),
        "std_x_coef": ("model", "std_x_coef"),
        "std_y_coef": ("model", "std_y_coef"),
        "gamma": ("algo", "gamma"),
        "gae_lambda": ("algo", "gae_lambda"),
    }
    if name not in mapping:
        return False
    section, key = mapping[name]
    algo_args.setdefault(section, {})[key] = value
    return True


def _apply_parameter_adjustments(algo_args, suggestions, safety_bounds=None):
    if safety_bounds is None:
        safety_bounds = {
            "lr": (1e-6, 1e-2),
            "critic_lr": (1e-6, 1e-2),
            "entropy_coef": (0.0, 0.1),
            "value_loss_coef": (0.1, 2.0),
            "std_x_coef": (0.1, 2.0),
            "std_y_coef": (0.1, 2.0),
            "gamma": (0.9, 0.999),
            "gae_lambda": (0.8, 0.99),
        }
    applied, rejected = {}, {}
    for name, new_value in suggestions.items():
        if name not in safety_bounds:
            rejected[name] = {"suggested": new_value, "reason": "unknown param"}
            continue
        lo, hi = safety_bounds[name]
        if not (lo <= new_value <= hi):
            rejected[name] = {"suggested": new_value, "bounds": (lo, hi), "reason": "out of bounds"}
            continue
        old_value = _get_param_value(algo_args, name)
        if _set_param_value(algo_args, name, new_value):
            applied[name] = {"old": old_value, "new": new_value}
        else:
            rejected[name] = {"suggested": new_value, "reason": "set failed"}
    return applied, rejected


def _apply_live_updates_to_runner(runner, accepted_updates):
    """Apply accepted hyperparameter updates to live runner objects.

    Note: simply mutating `runner.algo_args` is not enough because many hyperparameters
    are copied into actor/critic/buffer objects at initialization time.
    """
    if not accepted_updates:
        return

    def _set_optimizer_lr(optim, lr):
        try:
            for pg in optim.param_groups:
                pg["lr"] = float(lr)
        except Exception:
            pass

    # actor side (supports MAPPO and other on-policy actor algos using OnPolicyBase).
    if hasattr(runner, "actor") and runner.actor:
        # If parameter sharing is enabled, runner.actor entries may reference the same object.
        seen = set()
        for actor_algo in runner.actor:
            if id(actor_algo) in seen:
                continue
            seen.add(id(actor_algo))

            if "lr" in accepted_updates and hasattr(actor_algo, "actor_optimizer"):
                new_lr = float(accepted_updates["lr"])
                actor_algo.lr = new_lr
                if isinstance(getattr(actor_algo, "args", None), dict):
                    actor_algo.args["lr"] = new_lr
                _set_optimizer_lr(actor_algo.actor_optimizer, new_lr)

            if "entropy_coef" in accepted_updates and hasattr(actor_algo, "entropy_coef"):
                new_ent = float(accepted_updates["entropy_coef"])
                actor_algo.entropy_coef = new_ent
                if isinstance(getattr(actor_algo, "args", None), dict):
                    actor_algo.args["entropy_coef"] = new_ent

            # std_x/y live in the policy distribution (DiagGaussian) for continuous actions.
            if "std_x_coef" in accepted_updates or "std_y_coef" in accepted_updates:
                try:
                    action_out = actor_algo.actor.act.action_out
                except Exception:
                    action_out = None
                if action_out is not None:
                    if "std_x_coef" in accepted_updates and hasattr(action_out, "std_x_coef"):
                        action_out.std_x_coef = float(accepted_updates["std_x_coef"])
                    if "std_y_coef" in accepted_updates and hasattr(action_out, "std_y_coef"):
                        action_out.std_y_coef = float(accepted_updates["std_y_coef"])

    # critic side
    if hasattr(runner, "critic") and runner.critic is not None:
        critic = runner.critic

        if "critic_lr" in accepted_updates and hasattr(critic, "critic_optimizer"):
            new_clr = float(accepted_updates["critic_lr"])
            critic.critic_lr = new_clr
            if isinstance(getattr(critic, "args", None), dict):
                critic.args["critic_lr"] = new_clr
            _set_optimizer_lr(critic.critic_optimizer, new_clr)

        if "value_loss_coef" in accepted_updates and hasattr(critic, "value_loss_coef"):
            new_v = float(accepted_updates["value_loss_coef"])
            critic.value_loss_coef = new_v
            if isinstance(getattr(critic, "args", None), dict):
                critic.args["value_loss_coef"] = new_v

    # returns/advantage computation hyperparams live in buffers
    if hasattr(runner, "critic_buffer") and runner.critic_buffer is not None:
        cb = runner.critic_buffer
        if "gamma" in accepted_updates and hasattr(cb, "gamma"):
            cb.gamma = float(accepted_updates["gamma"])
        if "gae_lambda" in accepted_updates and hasattr(cb, "gae_lambda"):
            cb.gae_lambda = float(accepted_updates["gae_lambda"])


class LLMAutoAdjustTrainer:
    """LLM-guided auto adjustment trainer for on-policy runners."""

    def __init__(self, runner, llm_client, llm_model, guidance_interval=10, auto_adjust=True, safety_bounds=None):
        self.runner = runner
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.guidance_interval = guidance_interval
        self.auto_adjust = auto_adjust
        self.safety_bounds = safety_bounds
        self.parameter_history = []

    def run(self):
        runner = self.runner
        if runner.algo_args.get("render", {}).get("use_render", False):
            runner.run()
            return
        runner.warmup()

        episodes = (
            int(runner.algo_args["train"]["num_env_steps"])
            // runner.algo_args["train"]["episode_length"]
            // runner.algo_args["train"]["n_rollout_threads"]
        )

        if getattr(runner, "logger", None) is not None:
            runner.logger.init(episodes)

        for episode in range(1, episodes + 1):
            if runner.algo_args["train"].get("use_linear_lr_decay", False):
                if runner.share_param:
                    runner.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(runner.num_agents):
                        runner.actor[agent_id].lr_decay(episode, episodes)
                runner.critic.lr_decay(episode, episodes)

            if getattr(runner, "logger", None) is not None:
                runner.logger.episode_init(episode)

            runner.prep_rollout()
            for step in range(runner.algo_args["train"]["episode_length"]):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = runner.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = runner.envs.step(actions)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                if getattr(runner, "logger", None) is not None:
                    runner.logger.per_step(data)
                runner.insert(data)

            runner.compute()
            runner.prep_training()
            actor_train_infos, critic_train_info = runner.train()

            if self.auto_adjust and (episode % self.guidance_interval == 0):
                total_episodes = episodes
                env_meta = (
                    f"scenario={runner.env_args.get('scenario')}, "
                    f"agent_conf={runner.env_args.get('agent_conf')}, "
                    f"num_agents={getattr(runner, 'num_agents', None)}, "
                    f"episode={episode}/{total_episodes}"
                )
                if getattr(self, "stage_info", None):
                    env_meta += f", stage={self.stage_info}"
                ctx = {
                    "env_meta": env_meta,
                    "metrics": _format_training_context(episode, total_episodes, actor_train_infos, critic_train_info),
                    "safety_bounds": self.safety_bounds
                    if self.safety_bounds is not None
                    else {
                        "lr": [1e-6, 1e-2],
                        "critic_lr": [1e-6, 1e-2],
                        "entropy_coef": [0.0, 0.1],
                        "value_loss_coef": [0.1, 2.0],
                        "std_x_coef": [0.1, 2.0],
                        "std_y_coef": [0.1, 2.0],
                        "gamma": [0.9, 0.999],
                        "gae_lambda": [0.8, 0.99],
                    },
                }
                params = _format_current_params(runner.algo_args)
                guidance = _get_llm_guidance(self.llm_client, self.llm_model, ctx, params)
                suggestions = _extract_llm_adjustments(guidance)
                if suggestions:
                    applied, rejected = _apply_parameter_adjustments(
                        runner.algo_args, suggestions, safety_bounds=self.safety_bounds
                    )
                    accepted_updates = {k: v["new"] for k, v in applied.items()}
                    _apply_live_updates_to_runner(runner, accepted_updates)
                    if applied:
                        print(f"[LLM] Episode {episode}: applied {applied}")
                    if rejected:
                        print(f"[LLM] Episode {episode}: rejected {rejected}")
                    self.parameter_history.append(
                        {"episode": episode, "applied_changes": applied, "rejected_changes": rejected}
                    )

            if episode % runner.algo_args["train"]["log_interval"] == 0:
                if getattr(runner, "logger", None) is not None:
                    runner.logger.episode_log(
                        actor_train_infos, critic_train_info, runner.actor_buffer, runner.critic_buffer
                    )

            if episode % runner.algo_args["train"]["eval_interval"] == 0:
                if runner.algo_args["eval"]["use_eval"]:
                    runner.prep_rollout()
                    runner.eval()
                runner.save()

            runner.after_update()

        runner.save()


def run_curriculum(args, algo_args, env_args, *, llm_enabled=False, llm_client=None, llm_model=None, llm_guidance_interval=10):
    """Run curriculum training when curriculum_learning is enabled in env config."""
    import copy
    curriculum_cfg = env_args.get("curriculum_learning", {})
    stages = curriculum_cfg.get("stages", [])
    prev_models_dir = None
    prev_num_agents = None

    from harl.runners import RUNNER_REGISTRY

    for stage_idx, stage in enumerate(stages):
        agent_conf = stage.get("agent_conf")
        num_env_steps = stage.get("num_env_steps", algo_args["train"]["num_env_steps"])
        load_from_previous = stage.get("load_from_previous", False)
        if agent_conf is None:
            continue

        stage_env_args = copy.deepcopy(env_args)
        stage_env_args["agent_conf"] = agent_conf
        stage_algo_args = copy.deepcopy(algo_args)
        stage_algo_args["train"]["num_env_steps"] = num_env_steps

        if load_from_previous and prev_models_dir:
            stage_algo_args["train"]["model_dir"] = prev_models_dir
            stage_algo_args["train"]["curriculum_restore"] = True
            stage_algo_args["train"]["curriculum_source_num_agents"] = prev_num_agents
            print(f"\n[Curriculum] Stage {stage_idx + 1}: agent_conf={agent_conf}, loading from {prev_models_dir}")
        else:
            stage_algo_args["train"]["model_dir"] = None
            stage_algo_args["train"]["curriculum_restore"] = False
            print(f"\n[Curriculum] Stage {stage_idx + 1}: agent_conf={agent_conf} (from scratch)")

        runner = RUNNER_REGISTRY[args["algo"]](args, stage_algo_args, stage_env_args)
        if llm_enabled and _supports_llm_on_policy(runner):
            guided = LLMAutoAdjustTrainer(
                runner, llm_client, llm_model, guidance_interval=llm_guidance_interval, auto_adjust=True
            )
            guided.run()
        else:
            runner.run()
        prev_models_dir = runner.save_dir
        prev_num_agents = runner.num_agents
        runner.close()


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
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
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
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to env config yaml. Overrides default env config (e.g. for curriculum: harl/configs/envs_cfgs/mamujoco_curriculum_swimmer.yaml).",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Resume training in an existing run directory (reuse models/, append progress.txt).",
    )
    parser.add_argument(
        "--resume_add_steps",
        type=int,
        default=0,
        help="Additional env steps to train when resuming. New total = max(current_total, trained_steps + resume_add_steps).",
    )
    parser.add_argument(
        "--llm",
        type=str2bool,
        default=False,
        help="Enable LLM-guided auto adjustment (on-policy only). Use: --llm True/False",
    )
    parser.add_argument(
        "--curriculum",
        type=str2bool,
        default=None,
        help="Force enable/disable curriculum. If omitted, follows env config. Use: --curriculum True/False",
    )
    parser.add_argument(
        "--llm_guidance_interval",
        type=int,
        default=10,
        help="Interval (in episodes) between LLM guidance calls (when --llm True).",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    # Resume: if resume_dir is provided and load_config is not, auto-load its config.json.
    if args.get("resume_dir") and args.get("load_config", "") == "":
        import os

        cfg_path = os.path.join(args["resume_dir"], "config.json")
        if os.path.exists(cfg_path):
            args["load_config"] = cfg_path
        else:
            print(f"[Resume] warning: cannot find {cfg_path}, will use default yaml configs.")

    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    # Override env config from --config if provided (e.g. for curriculum)
    if args.get("config"):
        import yaml
        with open(args["config"], "r", encoding="utf-8") as f:
            env_override = yaml.load(f, Loader=yaml.FullLoader)
        env_args.update(env_override)
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # Apply resume settings (reuse run dir + restore from models + append progress)
    if args.get("resume_dir"):
        import os

        resume_dir = args["resume_dir"]
        env_args["resume_dir"] = resume_dir
        models_dir = os.path.join(resume_dir, "models")
        if os.path.isdir(models_dir):
            algo_args["train"]["model_dir"] = models_dir
        else:
            print(f"[Resume] warning: models dir not found: {models_dir}")

        # Infer already-trained steps from progress.txt last line: "<total_num_steps>,<eval_reward>"
        progress_path = os.path.join(resume_dir, "progress.txt")
        trained_steps = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if lines:
                    trained_steps = int(lines[-1].split(",")[0])
            except Exception:
                trained_steps = 0
        algo_args["train"]["resume_steps"] = trained_steps

        # Extend total training steps if requested
        if int(args.get("resume_add_steps", 0) or 0) > 0:
            new_total = trained_steps + int(args["resume_add_steps"])
            algo_args["train"]["num_env_steps"] = max(int(algo_args["train"]["num_env_steps"]), int(new_total))
        else:
            # If current target is already reached, stop early with a clear hint.
            if int(algo_args["train"]["num_env_steps"]) <= trained_steps:
                raise ValueError(
                    f"Resume target already reached: num_env_steps={algo_args['train']['num_env_steps']} <= trained_steps={trained_steps}. "
                    f"Pass --resume_add_steps to continue training."
                )

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # 课程学习：若 env 配置中启用了 curriculum_learning，则执行多阶段训练
    curriculum_cfg = env_args.get("curriculum_learning", {})
    curriculum_enabled = (
        args["curriculum"]
        if args.get("curriculum") is not None
        else (curriculum_cfg.get("enabled", False) and bool(curriculum_cfg.get("stages")))
    )
    if curriculum_enabled and not curriculum_cfg.get("stages"):
        print("Curriculum requested but no `curriculum_learning.stages` found in env config; falling back to single-stage training.")
        curriculum_enabled = False

    llm_client, llm_model = (None, None)
    if args.get("llm", False):
        llm_client, llm_model = init_llm_client(env_args)
        if llm_client is not None:
            print("LLM enabled.")

    if curriculum_enabled:
        run_curriculum(
            args,
            algo_args,
            env_args,
            llm_enabled=bool(args.get("llm", False) and llm_client is not None),
            llm_client=llm_client,
            llm_model=llm_model,
            llm_guidance_interval=args.get("llm_guidance_interval", 10),
        )
        return

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    if args.get("llm", False) and llm_client is not None:
        if _supports_llm_on_policy(runner):
            guided = LLMAutoAdjustTrainer(
                runner,
                llm_client,
                llm_model,
                guidance_interval=args.get("llm_guidance_interval", 10),
                auto_adjust=True,
            )
            guided.run()
        else:
            print("LLM auto-adjust currently supports on-policy runners only; falling back to normal training.")
            runner.run()
    else:
        runner.run()
    runner.close()


if __name__ == "__main__":
    main()
