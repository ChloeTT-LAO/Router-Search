#!/usr/bin/env python3
"""
多模态RAG系统评估脚本
支持WebQA、MMQA等数据集评估
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import logging
from tqdm import tqdm
import numpy as np

# 导入自定义模块
from src.stepwise_extensions.multimodal_inference_engine import MultimodalInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalRAGEvaluator:
    """多模态RAG系统评估器"""

    def __init__(self,
                 model_path: str,
                 kb_config_path: str,
                 output_dir: str = "./evaluation_results"):

        self.model_path = model_path
        self.kb_config_path = kb_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载知识库配置
        with open(kb_config_path, 'r') as f:
            self.kb_config = json.load(f)

        # 初始化推理引擎
        self.inference_engine = MultimodalInferenceEngine(
            model_path=model_path,
            kb_config=self.kb_config
        )

        # 评估指标
        self.metrics = {
            'accuracy': 0.0,
            'f1_recall': 0.0,
            'search_efficiency': 0.0,
            'avg_search_rounds': 0.0
        }

    def evaluate_webqa(self,
                       webqa_file: str,
                       image_dir: str = None,
                       max_samples: int = 500) -> Dict[str, Any]:
        """评估WebQA数据集"""
        logger.info(f"Evaluating WebQA dataset: {webqa_file}")

        # 加载数据
        with open(webqa_file, 'r', encoding='utf-8') as f:
            webqa_data = [json.loads(line) for line in f]

        if max_samples > 0:
            webqa_data = webqa_data[:max_samples]

        results = []
        correct_answers = 0
        total_search_rounds = 0
        f1_scores = []

        for i, item in enumerate(tqdm(webqa_data, desc="Evaluating WebQA")):
            try:
                question = item.get('Q', '')
                ground_truth = item.get('A', '')
                image_ids = item.get('img_posFacts', [])

                # 加载图像（如果有）
                image = None
                if image_ids and image_dir:
                    image_path = os.path.join(image_dir, f"{image_ids[0]}.jpg")
                    if os.path.exists(image_path):
                        image = Image.open(image_path)

                # 执行推理
                result = self.inference_engine.generate_response(
                    question=question,
                    image=image
                )

                # 计算指标
                predicted_answer = result['final_answer']
                is_correct = self._calculate_accuracy(predicted_answer, ground_truth)
                f1_score = self._calculate_f1_recall(predicted_answer, ground_truth)

                correct_answers += is_correct
                total_search_rounds += result['total_search_rounds']
                f1_scores.append(f1_score)

                # 保存结果
                results.append({
                    'question_id': i,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'f1_score': f1_score,
                    'search_rounds': result['total_search_rounds'],
                    'reasoning_trajectory': result['reasoning_trajectory']
                })

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} samples. "
                                f"Current accuracy: {correct_answers / (i + 1):.3f}")

            except Exception as e:
                logger.warning(f"Error processing WebQA item {i}: {e}")
                continue

        # 计算最终指标
        webqa_metrics = {
            'accuracy': correct_answers / len(results) if results else 0.0,
            'f1_recall': np.mean(f1_scores) if f1_scores else 0.0,
            'avg_search_rounds': total_search_rounds / len(results) if results else 0.0,
            'total_samples': len(results)
        }

        # 保存结果
        output_file = self.output_dir / "webqa_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': webqa_metrics,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"WebQA evaluation completed. Results saved to {output_file}")
        return webqa_metrics

    def evaluate_mmqa(self,
                      mmqa_file: str,
                      image_dir: str = None,
                      max_samples: int = 500) -> Dict[str, Any]:
        """评估MMQA数据集"""
        logger.info(f"Evaluating MMQA dataset: {mmqa_file}")

        # 加载数据
        with open(mmqa_file, 'r', encoding='utf-8') as f:
            mmqa_data = [json.loads(line) for line in f]

        if max_samples > 0:
            mmqa_data = mmqa_data[:max_samples]

        results = []
        correct_answers = 0
        total_search_rounds = 0
        f1_scores = []

        for i, item in enumerate(tqdm(mmqa_data, desc="Evaluating MMQA")):
            try:
                question = item.get('question', '')
                ground_truth = item.get('answer', '')

                # 加载图像（如果有）
                image = None
                if 'image_path' in item and image_dir:
                    image_path = os.path.join(image_dir, item['image_path'])
                    if os.path.exists(image_path):
                        image = Image.open(image_path)

                # 执行推理
                result = self.inference_engine.generate_response(
                    question=question,
                    image=image
                )

                # 计算指标
                predicted_answer = result['final_answer']
                is_correct = self._calculate_accuracy(predicted_answer, ground_truth)
                f1_score = self._calculate_f1_recall(predicted_answer, ground_truth)

                correct_answers += is_correct
                total_search_rounds += result['total_search_rounds']
                f1_scores.append(f1_score)

                # 保存结果
                results.append({
                    'question_id': i,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'f1_score': f1_score,
                    'search_rounds': result['total_search_rounds'],
                    'reasoning_trajectory': result['reasoning_trajectory']
                })

            except Exception as e:
                logger.warning(f"Error processing MMQA item {i}: {e}")
                continue

        # 计算最终指标
        mmqa_metrics = {
            'accuracy': correct_answers / len(results) if results else 0.0,
            'f1_recall': np.mean(f1_scores) if f1_scores else 0.0,
            'avg_search_rounds': total_search_rounds / len(results) if results else 0.0,
            'total_samples': len(results)
        }

        # 保存结果
        output_file = self.output_dir / "mmqa_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': mmqa_metrics,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"MMQA evaluation completed. Results saved to {output_file}")
        return mmqa_metrics

    def evaluate_infoseek(self,
                          infoseek_file: str,
                          image_dir: str,
                          max_samples: int = 500) -> Dict[str, Any]:
        """评估InfoSeek数据集"""
        logger.info(f"Evaluating InfoSeek dataset: {infoseek_file}")

        # 加载数据
        with open(infoseek_file, 'r', encoding='utf-8') as f:
            infoseek_data = [json.loads(line) for line in f]

        if max_samples > 0:
            infoseek_data = infoseek_data[:max_samples]

        results = []
        correct_answers = 0
        total_search_rounds = 0
        f1_scores = []

        for i, item in enumerate(tqdm(infoseek_data, desc="Evaluating InfoSeek")):
            try:
                question = item.get('question', '')
                ground_truth = item.get('answer', '')
                image_id = item.get('image_id', '')

                # 加载图像
                image = None
                if image_id and image_dir:
                    image_path = os.path.join(image_dir, f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        image = Image.open(image_path)

                # 执行推理
                result = self.inference_engine.generate_response(
                    question=question,
                    image=image
                )

                # 计算指标
                predicted_answer = result['final_answer']
                is_correct = self._calculate_accuracy(predicted_answer, ground_truth)
                f1_score = self._calculate_f1_recall(predicted_answer, ground_truth)

                correct_answers += is_correct
                total_search_rounds += result['total_search_rounds']
                f1_scores.append(f1_score)

                # 保存结果
                results.append({
                    'question_id': i,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'f1_score': f1_score,
                    'search_rounds': result['total_search_rounds']
                })

            except Exception as e:
                logger.warning(f"Error processing InfoSeek item {i}: {e}")
                continue

        # 计算最终指标
        infoseek_metrics = {
            'accuracy': correct_answers / len(results) if results else 0.0,
            'f1_recall': np.mean(f1_scores) if f1_scores else 0.0,
            'avg_search_rounds': total_search_rounds / len(results) if results else 0.0,
            'total_samples': len(results)
        }

        # 保存结果
        output_file = self.output_dir / "infoseek_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': infoseek_metrics,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"InfoSeek evaluation completed. Results saved to {output_file}")
        return infoseek_metrics

    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> bool:
        """计算精确匹配准确率"""
        if not predicted or not ground_truth:
            return False

        # 简单的字符串匹配
        predicted_clean = predicted.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()

        return predicted_clean == ground_truth_clean

    def _calculate_f1_recall(self, predicted: str, ground_truth: str) -> float:
        """计算F1-Recall分数"""
        if not predicted or not ground_truth:
            return 0.0

        # 分词
        predicted_tokens = set(predicted.lower().split())
        ground_truth_tokens = set(ground_truth.lower().split())

        if not ground_truth_tokens:
            return 0.0

        # 计算交集
        intersection = predicted_tokens.intersection(ground_truth_tokens)

        # 计算Recall (这里使用F1-Recall，实际上就是Recall)
        recall = len(intersection) / len(ground_truth_tokens)

        return recall

    def run_comprehensive_evaluation(self,
                                     datasets: Dict[str, str],
                                     image_dirs: Dict[str, str] = None,
                                     max_samples_per_dataset: int = 500) -> Dict[str, Any]:
        """运行综合评估"""
        logger.info("Starting comprehensive evaluation...")

        if image_dirs is None:
            image_dirs = {}

        all_results = {}

        # 评估WebQA
        if 'webqa' in datasets:
            webqa_results = self.evaluate_webqa(
                datasets['webqa'],
                image_dirs.get('webqa'),
                max_samples_per_dataset
            )
            all_results['webqa'] = webqa_results

        # 评估MMQA
        if 'mmqa' in datasets:
            mmqa_results = self.evaluate_mmqa(
                datasets['mmqa'],
                image_dirs.get('mmqa'),
                max_samples_per_dataset
            )
            all_results['mmqa'] = mmqa_results

        # 评估InfoSeek
        if 'infoseek' in datasets:
            infoseek_results = self.evaluate_infoseek(
                datasets['infoseek'],
                image_dirs.get('infoseek'),
                max_samples_per_dataset
            )
            all_results['infoseek'] = infoseek_results

        # 计算总体指标
        overall_metrics = self._calculate_overall_metrics(all_results)
        all_results['overall'] = overall_metrics

        # 保存综合结果
        output_file = self.output_dir / "comprehensive_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 打印结果
        self._print_results(all_results)

        logger.info(f"Comprehensive evaluation completed. Results saved to {output_file}")
        return all_results

    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算总体指标"""
        total_samples = 0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0
        weighted_search_rounds = 0.0

        for dataset_name, metrics in results.items():
            if dataset_name == 'overall':
                continue

            samples = metrics.get('total_samples', 0)
            if samples > 0:
                total_samples += samples
                weighted_accuracy += metrics.get('accuracy', 0.0) * samples
                weighted_f1 += metrics.get('f1_recall', 0.0) * samples
                weighted_search_rounds += metrics.get('avg_search_rounds', 0.0) * samples

        if total_samples > 0:
            return {
                'overall_accuracy': weighted_accuracy / total_samples,
                'overall_f1_recall': weighted_f1 / total_samples,
                'overall_avg_search_rounds': weighted_search_rounds / total_samples,
                'total_samples': total_samples
            }
        else:
            return {
                'overall_accuracy': 0.0,
                'overall_f1_recall': 0.0,
                'overall_avg_search_rounds': 0.0,
                'total_samples': 0
            }

    def _print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        for dataset_name, metrics in results.items():
            if dataset_name == 'overall':
                print(f"\n{'OVERALL PERFORMANCE':^60}")
                print("-" * 60)
            else:
                print(f"\n{dataset_name.upper() + ' DATASET':^60}")
                print("-" * 60)

            if 'accuracy' in metrics:
                print(f"Accuracy: {metrics['accuracy']:.4f}")
            if 'f1_recall' in metrics:
                print(f"F1-Recall: {metrics['f1_recall']:.4f}")
            if 'avg_search_rounds' in metrics:
                print(f"Avg Search Rounds: {metrics['avg_search_rounds']:.2f}")
            if 'total_samples' in metrics:
                print(f"Total Samples: {metrics['total_samples']}")

            # 打印overall指标
            if dataset_name == 'overall':
                for key, value in metrics.items():
                    if key.startswith('overall_') and key != 'total_samples':
                        metric_name = key.replace('overall_', '').replace('_', ' ').title()
                        print(f"{metric_name}: {value:.4f}")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate multimodal RAG system")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--kb_config", type=str, required=True,
                        help="Path to knowledge base configuration")
    parser.add_argument("--webqa_file", type=str,
                        help="Path to WebQA evaluation file")
    parser.add_argument("--mmqa_file", type=str,
                        help="Path to MMQA evaluation file")
    parser.add_argument("--infoseek_file", type=str,
                        help="Path to InfoSeek evaluation file")
    parser.add_argument("--webqa_image_dir", type=str,
                        help="Directory containing WebQA images")
    parser.add_argument("--mmqa_image_dir", type=str,
                        help="Directory containing MMQA images")
    parser.add_argument("--infoseek_image_dir", type=str,
                        help="Directory containing InfoSeek images")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum samples per dataset")

    args = parser.parse_args()

    # 初始化评估器
    evaluator = MultimodalRAGEvaluator(
        model_path=args.model_path,
        kb_config_path=args.kb_config,
        output_dir=args.output_dir
    )

    # 准备数据集
    datasets = {}
    image_dirs = {}

    if args.webqa_file:
        datasets['webqa'] = args.webqa_file
        if args.webqa_image_dir:
            image_dirs['webqa'] = args.webqa_image_dir

    if args.mmqa_file:
        datasets['mmqa'] = args.mmqa_file
        if args.mmqa_image_dir:
            image_dirs['mmqa'] = args.mmqa_image_dir

    if args.infoseek_file:
        datasets['infoseek'] = args.infoseek_file
        if args.infoseek_image_dir:
            image_dirs['infoseek'] = args.infoseek_image_dir

    # 运行评估
    if datasets:
        evaluator.run_comprehensive_evaluation(
            datasets=datasets,
            image_dirs=image_dirs,
            max_samples_per_dataset=args.max_samples
        )
    else:
        logger.error("No datasets specified for evaluation")


if __name__ == "__main__":
    main()