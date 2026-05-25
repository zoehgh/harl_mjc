"""Train an algorithm with LLM guidance that can automatically adjust parameters."""

import argparse
import json
import os
import re
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from openai import OpenAI

# LLM配置 - 从环境变量读取，如果没有则使用默认值
LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", "sk-063a3235cca746c7b1f8b49ca7223fa9"),
    "base_url": os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model": os.getenv("LLM_MODEL", "qwen3.5-plus")
}

def init_llm_client():
    """初始化LLM客户端"""
    try:
        client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )
        return client
    except Exception as e:
        print(f"LLM客户端初始化失败: {e}")
        return None

def get_llm_guidance(client, training_context, current_params):
    """获取LLM训练指导，并要求提供具体的参数调整建议"""
    if client is None:
        return "LLM客户端不可用，无法获取指导。"

    prompt = f"""
你是一个强化学习训练专家。请根据以下训练上下文，为Humanoid-v2环境的MAPPO算法训练提供指导建议。

训练上下文:
{training_context}

当前训练参数:
{current_params}

请提供具体的训练建议，包括：
1. 当前训练阶段的分析
2. 参数调整建议（必须包含具体的数值调整）
3. 训练策略建议

参数调整格式要求：
- 学习率调整：建议格式 "lr: 0.0005" 或 "lr: 0.0005 -> 0.0003"
- 熵系数调整：建议格式 "entropy_coef: 0.01" 或 "entropy_coef: 0.01 -> 0.005"
- 价值损失系数调整：建议格式 "value_loss_coef: 1.0 -> 0.8"
- 其他参数：类似格式

请保持建议简洁但具体。
"""

    try:
        completion = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[
                {'role': 'system', 'content': '你是一个专业的强化学习训练助手，擅长分析训练过程并提供具体的参数调整建议。'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM调用失败: {e}"

def format_training_context(episode, total_episodes, actor_train_infos, critic_train_info, eval_results=None):
    """格式化训练上下文信息"""
    context = f"""
当前训练进度: 第{episode}/{total_episodes}个训练周期

Actor训练信息:
"""
    for i, info in enumerate(actor_train_infos):
        context += f"Agent {i}: "
        if isinstance(info, dict):
            for key, value in info.items():
                context += f"{key}: {value:.4f}, "
        context += "\n"

    context += f"""
Critic训练信息:
"""
    if isinstance(critic_train_info, dict):
        for key, value in critic_train_info.items():
            context += f"{key}: {value:.4f}\n"

    if eval_results:
        context += f"""
评估结果:
{eval_results}
"""

    return context

def format_current_params(algo_args):
    """格式化当前训练参数"""
    params = f"""
学习率 (lr): {algo_args['model']['lr']}
Critic学习率 (critic_lr): {algo_args['model']['critic_lr']}
熵系数 (entropy_coef): {algo_args['algo']['entropy_coef']}
价值损失系数 (value_loss_coef): {algo_args['algo']['value_loss_coef']}
熵系数x (std_x_coef): {algo_args['model']['std_x_coef']}
熵系数y (std_y_coef): {algo_args['model']['std_y_coef']}
折扣因子 (gamma): {algo_args['algo']['gamma']}
GAE lambda (gae_lambda): {algo_args['algo']['gae_lambda']}
"""
    return params

def parse_llm_suggestions(llm_output):
    """解析LLM输出中的参数调整建议"""
    suggestions = {}

    # 定义参数映射 - 支持多种格式
    param_patterns = {
        'lr': [r'lr[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'lr[:：]\s*([\d.]+)'],
        'critic_lr': [r'critic_lr[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'critic_lr[:：]\s*([\d.]+)'],
        'entropy_coef': [r'entropy_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'entropy_coef[:：]\s*([\d.]+)'],
        'value_loss_coef': [r'value_loss_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'value_loss_coef[:：]\s*([\d.]+)'],
        'std_x_coef': [r'std_x_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'std_x_coef[:：]\s*([\d.]+)'],
        'std_y_coef': [r'std_y_coef[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'std_y_coef[:：]\s*([\d.]+)'],
        'gamma': [r'gamma[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'gamma[:：]\s*([\d.]+)'],
        'gae_lambda': [r'gae_lambda[:：]\s*[\d.]+\s*->\s*([\d.]+)', r'gae_lambda[:：]\s*([\d.]+)'],
    }

    for param_name, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    suggestions[param_name] = value
                    break  # 找到第一个匹配就停止
                except (ValueError, IndexError):
                    continue

    return suggestions

def apply_parameter_adjustments(algo_args, suggestions, safety_bounds=None):
    """应用参数调整建议，带安全边界检查"""
    if safety_bounds is None:
        safety_bounds = {
            'lr': (1e-6, 1e-2),
            'critic_lr': (1e-6, 1e-2),
            'entropy_coef': (0.0, 0.1),
            'value_loss_coef': (0.1, 2.0),
            'std_x_coef': (0.1, 2.0),
            'std_y_coef': (0.1, 2.0),
            'gamma': (0.9, 0.999),
            'gae_lambda': (0.8, 0.99),
        }

    applied_changes = {}
    rejected_changes = {}

    for param_name, new_value in suggestions.items():
        if param_name in safety_bounds:
            min_val, max_val = safety_bounds[param_name]
            # print(f"DEBUG: {param_name} 边界: [{min_val}, {max_val}], 值: {new_value}")
            if min_val <= new_value <= max_val:
                old_value = get_param_value(algo_args, param_name)
                # print(f"DEBUG: {param_name} 旧值: {old_value}")
                if set_param_value(algo_args, param_name, new_value):
                    applied_changes[param_name] = {'old': old_value, 'new': new_value}
                    print(f"✅ 应用参数调整: {param_name} = {old_value} -> {new_value}")
                else:
                    rejected_changes[param_name] = {
                        'suggested': new_value,
                        'reason': '设置失败'
                    }
                    print(f"❌ 参数设置失败: {param_name} = {new_value}")
            else:
                rejected_changes[param_name] = {
                    'suggested': new_value,
                    'bounds': (min_val, max_val),
                    'reason': f'超出安全边界 [{min_val}, {max_val}]'
                }
                print(f"❌ 拒绝参数调整: {param_name} = {new_value} (超出安全边界)")
        else:
            # print(f"DEBUG: {param_name} 不在安全边界中: {list(safety_bounds.keys())}")
            rejected_changes[param_name] = {
                'suggested': new_value,
                'reason': '未知参数'
            }
            print(f"❌ 拒绝参数调整: {param_name} = {new_value} (未知参数)")

    return applied_changes, rejected_changes

def get_nested_param(obj, param_path):
    """获取嵌套参数值"""
    keys = param_path.split('.')
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def set_nested_param(obj, param_path, value):
    """设置嵌套参数值"""
    keys = param_path.split('.')
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def get_param_value(algo_args, param_name):
    """根据参数名获取实际值"""
    param_mapping = {
        'lr': ('model', 'lr'),
        'critic_lr': ('model', 'critic_lr'),
        'entropy_coef': ('algo', 'entropy_coef'),
        'value_loss_coef': ('algo', 'value_loss_coef'),
        'std_x_coef': ('model', 'std_x_coef'),
        'std_y_coef': ('model', 'std_y_coef'),
        'gamma': ('algo', 'gamma'),
        'gae_lambda': ('algo', 'gae_lambda'),
    }

    if param_name in param_mapping:
        section, key = param_mapping[param_name]
        return algo_args.get(section, {}).get(key)
    return None

def set_param_value(algo_args, param_name, value):
    """根据参数名设置实际值"""
    param_mapping = {
        'lr': ('model', 'lr'),
        'critic_lr': ('model', 'critic_lr'),
        'entropy_coef': ('algo', 'entropy_coef'),
        'value_loss_coef': ('algo', 'value_loss_coef'),
        'std_x_coef': ('model', 'std_x_coef'),
        'std_y_coef': ('model', 'std_y_coef'),
        'gamma': ('algo', 'gamma'),
        'gae_lambda': ('algo', 'gae_lambda'),
    }

    if param_name in param_mapping:
        section, key = param_mapping[param_name]
        if section not in algo_args:
            algo_args[section] = {}
        algo_args[section][key] = value
        return True
    return False

class LLMAutoAdjustTrainer:
    """带有LLM自动参数调整的训练器"""

    def __init__(self, runner, llm_client, guidance_interval=10, auto_adjust=True, safety_bounds=None):
        self.runner = runner
        self.llm_client = llm_client
        self.guidance_interval = guidance_interval
        self.auto_adjust = auto_adjust
        # 如果没有提供safety_bounds，使用默认值
        if safety_bounds is None:
            self.safety_bounds = {
                'lr': (1e-6, 1e-2),
                'critic_lr': (1e-6, 1e-2),
                'entropy_coef': (0.0, 0.1),
                'value_loss_coef': (0.1, 2.0),
                'std_x_coef': (0.1, 2.0),
                'std_y_coef': (0.1, 2.0),
                'gamma': (0.9, 0.999),
                'gae_lambda': (0.8, 0.99),
            }
        else:
            self.safety_bounds = safety_bounds
        self.episode_count = 0
        self.parameter_history = []  # 记录参数调整历史

    def train_with_auto_adjustment(self):
        """带LLM自动参数调整的训练过程"""
        print("开始带有LLM自动参数调整的训练...")
        print("="*80)

        # 获取训练参数
        episodes = (
            int(self.runner.algo_args["train"]["num_env_steps"])
            // self.runner.algo_args["train"]["episode_length"]
            // self.runner.algo_args["train"]["n_rollout_threads"]
        )

        if hasattr(self.runner, 'logger') and self.runner.logger is not None:
            self.runner.logger.init(episodes)

        for episode in range(1, episodes + 1):
            self.episode_count = episode

            # 正常的训练步骤
            if self.runner.algo_args["train"]["use_linear_lr_decay"]:
                if self.runner.share_param:
                    self.runner.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.runner.num_agents):
                        self.runner.actor[agent_id].lr_decay(episode, episodes)
                self.runner.critic.lr_decay(episode, episodes)

            # logger callback at the beginning of each episode
            if hasattr(self.runner, 'logger') and self.runner.logger is not None:
                self.runner.logger.episode_init(episode)

            self.runner.prep_rollout()

            for step in range(self.runner.algo_args["train"]["episode_length"]):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.runner.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.runner.envs.step(actions)

                data = (
                    obs, share_obs, rewards, dones, infos, available_actions,
                    values, actions, action_log_probs, rnn_states, rnn_states_critic
                )

                # logger callback at each step
                if hasattr(self.runner, 'logger') and self.runner.logger is not None:
                    self.runner.logger.per_step(data)

                self.runner.insert(data)

            self.runner.compute()
            self.runner.prep_training()

            actor_train_infos, critic_train_info = self.runner.train()

            # LLM指导和自动参数调整逻辑
            if episode % self.guidance_interval == 0 and self.auto_adjust:
                applied_changes, rejected_changes = self._get_llm_guidance_and_adjust(
                    actor_train_infos, critic_train_info
                )

                # 记录参数调整历史
                if applied_changes:
                    self.parameter_history.append({
                        'episode': episode,
                        'applied_changes': applied_changes,
                        'rejected_changes': rejected_changes
                    })

            if episode % self.runner.algo_args["train"]["log_interval"] == 0:
                if hasattr(self.runner, 'logger') and self.runner.logger is not None:
                    self.runner.logger.episode_log(
                        actor_train_infos, critic_train_info,
                        self.runner.actor_buffer, self.runner.critic_buffer
                    )

            if episode % self.runner.algo_args["train"]["eval_interval"] == 0:
                if self.runner.algo_args["eval"]["use_eval"]:
                    self.runner.prep_rollout()
                    eval_results = self._run_evaluation()
                    self.runner.eval()
                self.runner.save()

            self.runner.after_update()

    def _run_evaluation(self):
        """运行评估并返回结果摘要"""
        return "评估完成，结果已记录到日志中。"

    def _get_llm_guidance_and_adjust(self, actor_train_infos, critic_train_info):
        """获取LLM指导并自动调整参数"""
        print("\n" + "="*80)
        print(f"🎯 第{self.episode_count}周期 - LLM自动参数调整")
        print("="*80)

        # 准备训练上下文
        training_context = format_training_context(
            self.episode_count,
            int(self.runner.algo_args["train"]["num_env_steps"])
            // self.runner.algo_args["train"]["episode_length"]
            // self.runner.algo_args["train"]["n_rollout_threads"],
            actor_train_infos,
            critic_train_info
        )

        # 准备当前参数信息
        current_params = format_current_params(self.runner.algo_args)

        print("📝 LLM输入上下文:")
        print("-" * 40)
        print(training_context)
        print("-" * 40)

        print("⚙️ 当前训练参数:")
        print("-" * 40)
        print(current_params)
        print("-" * 40)

        # 获取LLM指导
        guidance = get_llm_guidance(self.llm_client, training_context, current_params)

        print("🤖 LLM输出指导:")
        print("-" * 40)
        print(guidance)
        print("-" * 40)

        # 解析并应用参数调整
        suggestions = parse_llm_suggestions(guidance)
        if suggestions:
            print("🔧 解析到的参数调整建议:")
            for param, value in suggestions.items():
                print(f"  {param}: {value}")
            print("-" * 40)

            applied_changes, rejected_changes = apply_parameter_adjustments(
                self.runner.algo_args, suggestions, self.safety_bounds
            )

            print("-" * 40)
            if applied_changes:
                print("✅ 参数调整已应用到下次训练迭代")
            if rejected_changes:
                print("⚠️ 部分参数调整被安全机制拒绝")
        else:
            print("🔧 未检测到具体的参数调整建议")
            applied_changes, rejected_changes = {}, {}

        print("="*80 + "\n")

        return applied_changes, rejected_changes

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
        "--exp_name", type=str, default="llm_auto_adjust_training", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--llm_guidance_interval",
        type=int,
        default=10,
        help="Interval (in episodes) for LLM guidance requests.",
    )
    parser.add_argument(
        "--auto_adjust",
        action="store_true",
        default=True,
        help="Enable automatic parameter adjustment based on LLM suggestions.",
    )
    parser.add_argument(
        "--no_auto_adjust",
        action="store_true",
        default=False,
        help="Disable automatic parameter adjustment.",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=None,
        help="Number of environment steps to train for. If not specified, uses config file value.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to env config yaml (e.g. mamujoco_curriculum_swimmer.yaml for curriculum+LLM).",
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
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    # Override env config from --config (e.g. for curriculum)
    if args.get("config"):
        import yaml
        with open(args["config"], "r", encoding="utf-8") as f:
            env_override = yaml.load(f, Loader=yaml.FullLoader)
        env_args.update(env_override)
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # Override num_env_steps if specified via command line
    if args["num_env_steps"] is not None:
        algo_args["train"]["num_env_steps"] = args["num_env_steps"]

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # 固定使用非渲染模式以获得最佳性能
    print("📦 使用高性能egl后端（无渲染）")

    # 初始化LLM客户端
    llm_client = init_llm_client()
    if llm_client:
        print("✅ LLM客户端初始化成功")
    else:
        print("❌ LLM客户端初始化失败，将继续正常训练")

    from harl.runners import RUNNER_REGISTRY
    auto_adjust = args["auto_adjust"] and not args["no_auto_adjust"]

    # 课程学习 + LLM：若启用了 curriculum_learning，则按阶段训练，每阶段使用 LLM 自动调参
    curriculum_cfg = env_args.get("curriculum_learning", {})
    if curriculum_cfg.get("enabled", False) and curriculum_cfg.get("stages"):
        import copy
        prev_models_dir = None
        prev_num_agents = None
        for stage_idx, stage in enumerate(curriculum_cfg["stages"]):
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
                print(f"\n[Curriculum+LLM] Stage {stage_idx + 1}: agent_conf={agent_conf}, 加载上一阶段模型")
            else:
                stage_algo_args["train"]["model_dir"] = None
                stage_algo_args["train"]["curriculum_restore"] = False
                print(f"\n[Curriculum+LLM] Stage {stage_idx + 1}: agent_conf={agent_conf} (从头训练)")
            # 确保非渲染模式（训练时必须有 critic、save_dir 等）
            stage_algo_args["render"]["use_render"] = False
            runner = RUNNER_REGISTRY[args["algo"]](args, stage_algo_args, stage_env_args)
            guided_trainer = LLMAutoAdjustTrainer(
                runner, llm_client, args["llm_guidance_interval"], auto_adjust=auto_adjust
            )
            try:
                guided_trainer.train_with_auto_adjustment()
            finally:
                prev_models_dir = getattr(runner, "save_dir", None) or (
                    os.path.join(runner.run_dir, "models") if getattr(runner, "run_dir", None) else None
                )
                prev_num_agents = getattr(runner, "num_agents", None)
                try:
                    runner.close()
                except Exception:
                    pass
        print("\n课程学习+LLM 训练完成")
        return

    # 单阶段训练 + LLM
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    guided_trainer = LLMAutoAdjustTrainer(
        runner, llm_client, args["llm_guidance_interval"], auto_adjust=auto_adjust
    )
    try:
        guided_trainer.train_with_auto_adjustment()
        print(f"\n参数调整历史记录: {len(guided_trainer.parameter_history)} 次调整")
        for record in guided_trainer.parameter_history[-3:]:
            print(f"周期 {record['episode']}: {record['applied_changes']}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
    finally:
        runner.close()
        print("训练结束")


if __name__ == "__main__":
    main()