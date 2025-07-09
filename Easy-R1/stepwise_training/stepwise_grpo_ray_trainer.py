#!/usr/bin/env python3
"""
StepSearch风格的GRPO训练器
基于R1-Router的ray_trainer.py架构，集成StepSearch的逐步奖励机制
"""

import os
import json
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type, List

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer, ProcessorMixin

# 导入原有的ray trainer组件
try:
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.single_controller.base import Worker
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer import core_algos
    from verl.trainer.config import PPOConfig
    from verl.utils.rl_dataset import RLHFDataset, collate_fn
    from verl.utils.torch_functional import masked_mean
    from verl.utils.tracking import Tracking
    from verl.workers.fsdp_workers import FSDPWorker
except ImportError:
    print("警告: 无法导入VERL组件，请确保Easy-R1正确安装")

# 导入自定义模块
from examples.stepwise_multimodal_rewards import StepwiseMultimodalReward
from multimodal_search_tools import MultimodalSearchTools

WorkerType = Type[Worker]


class Role(Enum):
    """扩展Role类以支持多模态检索"""
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    # 新增多模态相关角色
    MultimodalRetriever = 7


@dataclass
class StepwiseResourcePoolManager:
    """StepSearch风格的资源池管理器"""
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    # StepSearch特定配置
    enable_stepwise_rewards: bool = True
    enable_multimodal_search: bool = True
    kb_config_path: str = ""

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            self.resource_pool_dict[resource_pool_name] = RayResourcePool(
                resource_pool_name=resource_pool_name,
                process_on_nodes=process_on_nodes,
                max_colocate_count=1  # FSDP推荐设置
            )


class StepwiseRayPPOTrainer:
    """
    StepSearch风格的Ray PPO训练器
    基于原有ray_trainer.py架构，集成逐步奖励机制
    """

    def __init__(
            self,
            config: PPOConfig,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            role_worker_mapping: dict[Role, WorkerType],
            resource_pool_manager: StepwiseResourcePoolManager,
            ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            # StepSearch特定参数
            stepwise_reward_config: Dict[str, float] = None,
            kb_config_path: str = "",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # StepSearch配置
        self.stepwise_reward_config = stepwise_reward_config or {
            'information_gain_weight': 0.3,
            'redundancy_penalty_weight': 0.2,
            'routing_accuracy_weight': 0.3,
            'answer_quality_weight': 0.2
        }
        self.kb_config_path = kb_config_path

        # 初始化StepSearch组件
        self._init_stepwise_components()

        self.hybrid_engine = config.worker.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # KL控制器设置
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        # 优势估计器设置 - 强制使用GRPO
        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        else:
            raise ValueError(f"Unknown adv_estimator: {self.config.algorithm.adv_estimator}")

        # 初始化worker groups
        self._init_worker_groups()

        # 设置数据加载器
        self._setup_dataloader()

        self.global_steps = 0

    def _init_stepwise_components(self):
        """初始化StepSearch组件"""
        # 初始化奖励计算器
        self.stepwise_reward_calculator = StepwiseMultimodalReward(
            alpha_gain=self.stepwise_reward_config['information_gain_weight'],
            alpha_redundancy=self.stepwise_reward_config['redundancy_penalty_weight'],
            alpha_route=self.stepwise_reward_config['routing_accuracy_weight'],
            alpha_answer=self.stepwise_reward_config['answer_quality_weight']
        )

        # 初始化多模态搜索工具
        if self.kb_config_path and os.path.exists(self.kb_config_path):
            with open(self.kb_config_path, 'r') as f:
                kb_config = json.load(f)
            self.search_tools = MultimodalSearchTools(kb_config)
        else:
            self.search_tools = None
            print("警告: 知识库配置文件不存在，禁用搜索功能")

    def _init_worker_groups(self):
        """初始化worker groups - 基于原有实现"""
        resource_pool_manager = self.resource_pool_manager
        resource_pool_manager.create_resource_pool()

        # 构建资源池到类的映射
        self.resource_pool_to_cls = {}
        for role, worker_cls in self.role_worker_mapping.items():
            resource_pool_name = resource_pool_manager.mapping[role]
            if resource_pool_name not in self.resource_pool_to_cls:
                self.resource_pool_to_cls[resource_pool_name] = {}

            # 为不同角色设置不同的key名称
            role_key_mapping = {
                Role.ActorRollout: "actor_rollout",
                Role.Critic: "critic",
                Role.RefPolicy: "ref",
                Role.RewardModel: "rm",
                Role.MultimodalRetriever: "retriever"
            }
            key_name = role_key_mapping.get(role, role.name.lower())
            self.resource_pool_to_cls[resource_pool_name][key_name] = worker_cls

        # 创建worker groups
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool_manager.resource_pool_dict[resource_pool],
                ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        # 初始化各个worker group
        if self.use_critic:
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        # 最后创建actor rollout
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # 如果有多模态检索器
        if "retriever" in all_wg:
            self.retriever_wg = all_wg["retriever"]
            self.retriever_wg.init_model()

    def _setup_dataloader(self):
        """设置数据加载器"""
        # 训练数据集
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=self.config.data.system_prompt,
            min_pixels=getattr(self.config.data, 'min_pixels', None),
            max_pixels=getattr(self.config.data, 'max_pixels', None),
        )

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # 验证数据集
        if hasattr(self.config.data, 'val_files') and self.config.data.val_files:
            self.val_dataset = RLHFDataset(
                parquet_files=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                system_prompt=self.config.data.system_prompt,
                min_pixels=getattr(self.config.data, 'min_pixels', None),
                max_pixels=getattr(self.config.data, 'max_pixels', None),
            )

            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=len(self.val_dataset) // 2,
                num_workers=8,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            self.val_dataloader = None

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        if self.val_dataloader:
            print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # 计算总训练步数
        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        if self.use_critic:
            self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def compute_stepwise_advantages(self, data: DataProto) -> DataProto:
        """
        计算StepSearch风格的优势值
        融合原有GRPO逻辑和StepSearch的逐步奖励
        """
        # 获取基础信息
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        # 基础GRPO优势计算
        index = data.non_tensor_batch.get("uid", None)
        if index is not None:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index
            )
        else:
            # 如果没有uid，使用标准的REINFORCE优势
            advantages = token_level_rewards
            returns = token_level_rewards

        # StepSearch逐步奖励增强
        if hasattr(data.non_tensor_batch, 'reasoning_steps'):
            stepwise_rewards = self._compute_stepwise_rewards(data)
            # 将逐步奖励加入到优势计算中
            advantages = advantages + 0.1 * stepwise_rewards  # 0.1是加权因子

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        return data

    def _compute_stepwise_rewards(self, data: DataProto) -> torch.Tensor:
        """计算StepSearch风格的逐步奖励"""
        batch_size = data.batch["responses"].size(0)
        response_length = data.batch["responses"].size(-1)

        # 初始化逐步奖励张量
        stepwise_rewards = torch.zeros(batch_size, response_length, device=data.batch["responses"].device)

        # 为每个样本计算逐步奖励
        for i in range(batch_size):
            if 'reasoning_steps' in data.non_tensor_batch:
                reasoning_steps = data.non_tensor_batch['reasoning_steps'][i]
                search_history = []

                # 计算每个推理步骤的奖励
                step_rewards = []
                for step in reasoning_steps:
                    # 模拟搜索结果
                    search_results = self._simulate_search_for_step(step)

                    # 计算逐步奖励
                    step_reward_dict = self.stepwise_reward_calculator.compute_step_reward(
                        step_data=step,
                        search_results=search_results,
                        search_history=search_history
                    )

                    step_rewards.append(step_reward_dict['total_step_reward'])

                    # 更新搜索历史
                    search_history.append({
                        'query': step.get('sub_question', ''),
                        'search_results': search_results,
                        'step_reward': step_reward_dict['total_step_reward']
                    })

                # 将步骤奖励分配到token级别
                if step_rewards:
                    avg_step_reward = np.mean(step_rewards)
                    stepwise_rewards[i, :] = avg_step_reward

        return stepwise_rewards

    def _simulate_search_for_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """为训练步骤模拟搜索结果"""
        # 在实际实现中，这里应该调用真实的搜索系统
        # 为了训练，我们提供模拟结果
        query = step.get('sub_question', '')
        retriever_type = step.get('selected_retriever', 'Text Retriever')

        if not query:
            return {'results': [], 'search_type': 'none'}

        # 如果有搜索工具，使用真实搜索
        if self.search_tools:
            try:
                return self.search_tools.execute_search(query, retriever_type, top_k=3)
            except Exception as e:
                print(f"搜索失败: {e}")

        # 模拟搜索结果
        mock_results = [
            {
                'content': f"Mock search result for query: {query}",
                'score': 0.8,
                'source': 'mock_kb'
            }
        ]

        return {
            'results': mock_results,
            'search_type': retriever_type.lower().replace(' ', '_'),
            'query': query
        }

    def _save_checkpoint(self):
        """保存检查点"""
        local_global_step_folder = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_local_path,
                self.global_steps,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.save_checkpoint_path, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """加载检查点"""
        if self.config.trainer.load_checkpoint_path is None:
            return

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}")
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

    def _validate(self):
        """验证模型性能"""
        if self.val_reward_fn is None or self.val_dataloader is None:
            return {}

        self.actor_rollout_wg.eval()
        val_metrics = {}

        for batch_dict in self.val_dataloader:
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # 执行验证生成
            if "pixel_values" in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            # 计算验证奖励
            val_rewards = self.val_reward_fn(gen_batch_output)
            val_metrics["val_reward/mean"] = torch.mean(val_rewards).item()
            break  # 只验证一个batch

        self.actor_rollout_wg.train()
        return val_metrics

    def fit(self):
        """
        StepSearch风格的训练循环
        基于原有ray_trainer.py的fit()方法，集成逐步奖励机制
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        self.global_steps = 0

        # 加载检查点
        self._load_checkpoint()

        # 训练前验证
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.val_only:
                return

        # 主训练循环
        for _ in range(self.config.trainer.total_episodes):
            for batch_dict in self.train_dataloader:
                self.global_steps += 1
                if self.global_steps >= self.training_steps:
                    break

                metrics = {}
                timing_raw = {}

                print(f"Training step: {self.global_steps}/{self.training_steps}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 准备生成batch
                if "pixel_values" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with _timer("step", timing_raw):
                    # 生成序列
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # 计算参考策略logprobs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_batch = self.ref_policy_wg.generate_sequences(gen_batch_output)
                            gen_batch_output.batch["ref_log_probs"] = ref_batch.batch["log_probs"]

                    # 计算奖励
                    with _timer("reward", timing_raw):
                        rewards = self.reward_fn(gen_batch_output)
                        gen_batch_output.batch["token_level_rewards"] = rewards

                    # 如果使用critic，计算values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values_batch = self.critic_wg.generate_sequences(gen_batch_output)
                            gen_batch_output.batch["values"] = values_batch.batch["values"]

                    # StepSearch风格的优势计算
                    with _timer("adv", timing_raw):
                        gen_batch_output = self.compute_stepwise_advantages(gen_batch_output)

                    # 更新critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_metrics = self.critic_wg.update_critic(gen_batch_output)
                            metrics.update(critic_metrics)

                    # 更新actor
                    with _timer("update_actor", timing_raw):
                        actor_metrics = self.actor_rollout_wg.update_actor(gen_batch_output, self.kl_ctrl)
                        metrics.update(actor_metrics)

                    # 定期验证
                    if (self.config.trainer.val_freq > 0 and
                            self.global_steps % self.config.trainer.val_freq == 0 and
                            self.val_reward_fn is not None):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                        metrics.update(val_metrics)

                    # 定期保存
                    if (self.config.trainer.save_freq > 0 and
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # 收集指标
                metrics.update(self._compute_data_metrics(batch=gen_batch_output))
                metrics.update(self._compute_timing_metrics(batch=gen_batch_output, timing_raw=timing_raw))

                # 添加StepSearch特定指标
                if hasattr(gen_batch_output.batch, 'stepwise_rewards'):
                    metrics["stepwise/reward_mean"] = torch.mean(gen_batch_output.batch['stepwise_rewards']).item()

                # 记录日志
                logger.log(data=metrics, step=self.global_steps)

        # 训练后验证
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            pprint(f"Final validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        # 保存最终检查点
        self._save_checkpoint()

    def _compute_data_metrics(self, batch: DataProto) -> Dict[str, float]:
        """计算数据指标"""
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]

        response_length = responses.size(-1)
        response_mask = attention_mask[:, -response_length:]
        prompt_length = attention_mask.size(-1) - response_length

        max_response_length = self.config.data.max_response_length
        max_prompt_length = self.config.data.max_prompt_length

        actual_response_length = torch.sum(response_mask, dim=-1)
        actual_prompt_length = torch.full_like(actual_response_length, prompt_length)

        return {
            "response_length/mean": torch.mean(actual_response_length.float()).item(),
            "response_length/max": torch.max(actual_response_length).item(),
            "response_length/min": torch.min(actual_response_length).item(),
            "response_length/clip_ratio": torch.mean(
                torch.eq(actual_response_length, max_response_length).float()
            ).item(),
            "prompt_length/mean": torch.mean(actual_prompt_length.float()).item(),
            "prompt_length/max": torch.max(actual_prompt_length).item(),
            "prompt_length/min": torch.min(actual_prompt_length).item(),
            "prompt_length/clip_ratio": torch.mean(
                torch.eq(actual_prompt_length, max_prompt_length).float()
            ).item(),
        }

    def _compute_timing_metrics(self, batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, float]:
        """计算时间指标"""
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]

        response_length = responses.size(-1)
        response_mask = attention_mask[:, -response_length:]

        num_prompt_tokens = torch.sum(attention_mask).item() - torch.sum(response_mask).item()
        num_response_tokens = torch.sum(response_mask).item()
        num_overall_tokens = num_prompt_tokens + num_response_tokens

        num_tokens_of_section = {
            "gen": num_response_tokens,
            **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
        }

        return {
            **{f"timing_s/{name}": value for name, value in timing_raw.items()},
            **{
                f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
                for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
            },
        }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """计时器上下文管理器"""
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def create_stepwise_trainer(
        config_path: str,
        stepwise_reward_config: Dict[str, float],
        kb_config_path: str
) -> StepwiseRayPPOTrainer:
    """创建StepSearch风格的训练器实例"""

    # 加载配置
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 这里需要根据实际配置结构调整
    config = PPOConfig(**config_dict)

    # 初始化tokenizer和processor
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    processor = AutoProcessor.from_pretrained(config.model.name_or_path)

    # 定义角色到worker的映射
    role_worker_mapping = {
        Role.ActorRollout: FSDPWorker,
        Role.Critic: FSDPWorker,
        Role.RefPolicy: FSDPWorker,
        # 根据需要添加其他角色
    }

    # 资源池管理器
    resource_pool_manager = StepwiseResourcePoolManager(
        resource_pool_spec={"default": [0, 1, 2, 3]},  # 使用GPU 0-3
        mapping={
            Role.ActorRollout: "default",
            Role.Critic: "default",
            Role.RefPolicy: "default",
        },
        enable_stepwise_rewards=True,
        enable_multimodal_search=True,
        kb_config_path=kb_config_path
    )

    # 创建训练器
    trainer = StepwiseRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        stepwise_reward_config=stepwise_reward_config,
        kb_config_path=kb_config_path
    )

    return trainer


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="StepSearch风格的GRPO训练")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径")
    parser.add_argument("--kb_config", type=str, required=True, help="知识库配置文件路径")
    parser.add_argument("--info_gain_weight", type=float, default=0.3, help="信息增益权重")
    parser.add_argument("--redundancy_weight", type=float, default=0.2, help="冗余惩罚权重")
    parser.add_argument("--routing_weight", type=float, default=0.3, help="路由准确性权重")
    parser.add_argument("--answer_weight", type=float, default=0.2, help="答案质量权重")

    args = parser.parse_args()

    # StepSearch奖励配置
    stepwise_reward_config = {
        'information_gain_weight': args.info_gain_weight,
        'redundancy_penalty_weight': args.redundancy_weight,
        'routing_accuracy_weight': args.routing_weight,
        'answer_quality_weight': args.answer_weight
    }

    # 创建训练器
    trainer = create_stepwise_trainer(
        config_path=args.config,
        stepwise_reward_config=stepwise_reward_config,
        kb_config_path=args.kb_config
    )

    # 开始训练
    print("开始StepSearch风格的GRPO训练...")
    trainer.fit()
    print("训练完成!")


if __name__ == "__main__":
    main()