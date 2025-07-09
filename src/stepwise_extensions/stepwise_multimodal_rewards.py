import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class StepwiseMultimodalReward:
    """
    融合StepSearch逐步奖励机制的多模态奖励计算器
    包含：信息增益奖励、冗余惩罚、路由准确性奖励、答案质量奖励
    """

    def __init__(self,
                 alpha_gain: float = 0.3,
                 alpha_redundancy: float = 0.2,
                 alpha_route: float = 0.3,
                 alpha_answer: float = 0.2,
                 alpha_format: float = 0.1):
        self.alpha_gain = alpha_gain
        self.alpha_redundancy = alpha_redundancy
        self.alpha_route = alpha_route
        self.alpha_answer = alpha_answer
        self.alpha_format = alpha_format

        self.sentence_encoder = SentenceTransformer('BAAI/bge-m3')

    def compute_step_reward(self,
                            step_data: Dict[str, Any],
                            search_results: Dict[str, Any],
                            search_history: List[Dict[str, Any]],
                            ground_truth: Dict[str, Any] = None) -> Dict[str, float]:
        """
        计算单步奖励，融合StepSearch的逐步监督机制

        Args:
            step_data: 当前步骤数据，包含query, selected_retriever等
            search_results: 搜索结果
            search_history: 搜索历史
            ground_truth: 真实标签数据

        Returns:
            包含各项奖励分数的字典
        """
        rewards = {}

        # 1. 信息增益奖励 (Information Gain Reward)
        info_gain = self._calculate_information_gain(search_results, search_history)
        rewards['information_gain'] = info_gain

        # 2. 冗余惩罚 (Redundancy Penalty)
        redundancy_penalty = self._calculate_redundancy_penalty(
            step_data.get('query', ''), search_history
        )
        rewards['redundancy_penalty'] = redundancy_penalty

        # 3. 路由准确性奖励 (Routing Accuracy Reward)
        if ground_truth and 'ground_truth_retriever' in ground_truth:
            routing_reward = self._calculate_routing_reward(
                step_data.get('selected_retriever', ''),
                ground_truth['ground_truth_retriever']
            )
            rewards['routing_accuracy'] = routing_reward
        else:
            rewards['routing_accuracy'] = 0.0

        # 4. 搜索查询质量奖励 (Search Query Quality Reward)
        if ground_truth and 'ground_truth_query' in ground_truth:
            query_reward = self._calculate_query_reward(
                step_data.get('query', ''),
                ground_truth['ground_truth_query']
            )
            rewards['query_quality'] = query_reward
        else:
            rewards['query_quality'] = 0.0

        # 5. 格式奖励 (Format Reward)
        format_reward = self._calculate_format_reward(step_data)
        rewards['format_reward'] = format_reward

        # 计算综合奖励
        total_reward = (
                self.alpha_gain * info_gain -
                self.alpha_redundancy * redundancy_penalty +
                self.alpha_route * rewards['routing_accuracy'] +
                0.1 * rewards['query_quality'] +
                self.alpha_format * format_reward
        )

        rewards['total_step_reward'] = total_reward

        return rewards

    def compute_final_answer_reward(self,
                                    generated_answer: str,
                                    ground_truth_answer: str,
                                    search_penalty: float = 0.0) -> Dict[str, float]:
        """
        计算最终答案奖励

        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 真实答案
            search_penalty: 搜索惩罚（基于搜索次数）

        Returns:
            奖励字典
        """
        rewards = {}

        # 答案准确性奖励
        accuracy_reward = self._calculate_answer_accuracy(
            generated_answer, ground_truth_answer
        )
        rewards['answer_accuracy'] = accuracy_reward

        # 搜索效率惩罚
        rewards['search_penalty'] = search_penalty

        # 最终奖励
        final_reward = accuracy_reward - search_penalty
        rewards['final_reward'] = final_reward

        return rewards

    def _calculate_information_gain(self,
                                    search_results: Dict[str, Any],
                                    search_history: List[Dict[str, Any]]) -> float:
        """计算信息增益 - StepSearch核心机制"""
        if not search_results or not search_results.get('results'):
            return 0.0

        # 提取新检索到的内容
        new_content = []
        for result in search_results['results']:
            content = result.get('content', '')
            if content.strip():
                new_content.append(content)

        if not new_content:
            return 0.0

        new_text = ' '.join(new_content)

        # 如果没有历史信息，则为完全新信息
        if not search_history:
            return 1.0

        # 提取历史检索内容
        historical_content = []
        for history_entry in search_history:
            if 'search_results' in history_entry:
                for result in history_entry['search_results'].get('results', []):
                    content = result.get('content', '')
                    if content.strip():
                        historical_content.append(content)

        if not historical_content:
            return 1.0

        historical_text = ' '.join(historical_content)

        # 计算新内容与历史内容的语义相似度
        try:
            embeddings = self.sentence_encoder.encode([new_text, historical_text])
            similarity = F.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            ).item()

            # 信息增益 = 1 - 相似度，确保范围在[0,1]
            information_gain = max(0.0, min(1.0, 1.0 - similarity))

        except Exception as e:
            logger.warning(f"Error calculating information gain: {e}")
            information_gain = 0.5  # 默认中等增益

        return information_gain

    def _calculate_redundancy_penalty(self,
                                      current_query: str,
                                      search_history: List[Dict[str, Any]]) -> float:
        """计算冗余惩罚 - StepSearch核心机制"""
        if not current_query.strip() or not search_history:
            return 0.0

        # 提取历史查询
        historical_queries = []
        for history_entry in search_history:
            query = history_entry.get('query', '')
            if query.strip():
                historical_queries.append(query)

        if not historical_queries:
            return 0.0

        # 计算当前查询与历史查询的最大相似度
        try:
            all_queries = [current_query] + historical_queries
            embeddings = self.sentence_encoder.encode(all_queries)

            current_embedding = torch.tensor(embeddings[0]).unsqueeze(0)
            historical_embeddings = torch.tensor(embeddings[1:])

            similarities = F.cosine_similarity(
                current_embedding.expand(len(historical_queries), -1),
                historical_embeddings,
                dim=1
            )

            max_similarity = similarities.max().item()
            redundancy_penalty = max(0.0, min(1.0, max_similarity))

        except Exception as e:
            logger.warning(f"Error calculating redundancy penalty: {e}")
            redundancy_penalty = 0.0

        return redundancy_penalty

    def _calculate_routing_reward(self,
                                  selected_retriever: str,
                                  ground_truth_retriever: str) -> float:
        """计算路由准确性奖励"""
        if not selected_retriever or not ground_truth_retriever:
            return 0.0

        # 精确匹配
        if selected_retriever.strip() == ground_truth_retriever.strip():
            return 1.0
        else:
            return 0.0

    def _calculate_query_reward(self,
                                generated_query: str,
                                ground_truth_query: str) -> float:
        """计算查询质量奖励"""
        if not generated_query.strip() or not ground_truth_query.strip():
            return 0.0

        try:
            embeddings = self.sentence_encoder.encode([generated_query, ground_truth_query])
            similarity = F.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            ).item()

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"Error calculating query reward: {e}")
            return 0.0

    def _calculate_format_reward(self, step_data: Dict[str, Any]) -> float:
        """计算格式奖励"""
        format_score = 0.0

        # 检查是否包含必要的字段
        if 'query' in step_data and step_data['query'].strip():
            format_score += 0.5

        if 'selected_retriever' in step_data and step_data['selected_retriever'].strip():
            format_score += 0.5

        return format_score

    def _calculate_answer_accuracy(self,
                                   generated_answer: str,
                                   ground_truth_answer: str) -> float:
        """计算答案准确性"""
        if not generated_answer.strip() or not ground_truth_answer.strip():
            return 0.0

        # 使用F1-Recall计算
        generated_tokens = set(generated_answer.lower().split())
        ground_truth_tokens = set(ground_truth_answer.lower().split())

        if not ground_truth_tokens:
            return 0.0

        intersection = generated_tokens.intersection(ground_truth_tokens)
        recall = len(intersection) / len(ground_truth_tokens)

        return recall

    def calculate_search_penalty(self, num_searches: int, max_searches: int = 5) -> float:
        """计算搜索效率惩罚"""
        if num_searches <= 1:
            return 0.0
        elif num_searches <= max_searches:
            return 0.1 * (num_searches - 1) / (max_searches - 1)
        else:
            return 0.2  # 超过最大搜索次数的额外惩罚