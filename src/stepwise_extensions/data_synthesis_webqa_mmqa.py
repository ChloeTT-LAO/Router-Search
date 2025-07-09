import json
import os
import argparse
import random
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedR1RouterSynthesizer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.max_reasoning_steps = 3

        # 初始化模型
        self._initialize_model()

        # 初始化检索器（可选，用于真实检索）
        self.enable_real_retrieval = False
        self.text_retriever = None
        self.image_retriever = None
        self.table_retriever = None

        # R1-Router式的系统提示词
        self.system_prompt = """You are a professional question decomposition expert for multi-hop QA systems. Your task is to decompose complex questions into strictly single-hop sub-questions and select appropriate retrievers.

Strict Output Format:
<think>Analyze the original question and determine the next required sub-question. Do NOT reveal answers or perform multi-hop reasoning.</think>
<sub-question>Exactly ONE single-hop question one time. If no further information is needed to answer the original question, write "None".</sub-question>
<ret>Choose 1 retriever from: Text Retriever, Text Image Retriever, Table Retriever. Write "None" if <sub-question> is "None".</ret>

Critical Rules:
1. Atomic Sub-question Definition:
   a) A sub-question is "atomic" only if:
   b) It cannot be further decomposed into simpler questions
   c) It requires exactly *one retrieval action* to answer
   d) Does NOT depend on answers to previous sub-questions
   e) It can be helpful to answer the origin question

2. Retriever Selection Guidelines:
   - Text Retriever: For non-visual commonsense knowledge (e.g., "Define photosynthesis")
   - Text Image Retriever: When sub-question explicitly references visual elements (e.g., "Describe the painting style of...")
   - Table Retriever: For numerical/statistical queries (e.g., "GDP of Japan in 2020")

3. Strict Prohibitions:
   - Never combine multiple questions in <sub-question>
   - Never mention retrieved content in <think>
   - Never select retrievers for non-atomic questions

Origin Question: {question}
{image_context}
Reasoning Trajectories: {current_reasoning_trajectories}"""

        # 中间答案生成提示词
        self.intermediate_answer_prompt = """You are a professional question answering model. Your task is to carefully think through the question based on the information retrieved and then provide the final answer.

Strict Output Format:
<think>Analyze the sub-question and the retrieved information. Break down the reasoning process step by step. Do NOT provide the final answer yet.</think>
<answer>Provide the answer to the sub-question based solely on the retrieved information.</answer>

Sub-question: {sub_question}
Retrieved Information: {retrieved_info}
{image_context}

Based on the above information, answer the sub-question."""

        # 最终答案生成提示词
        self.final_answer_prompt = """You are a professional question answering model. Your task is to carefully think through the question based on the sub-questions and its answers and then provide the final answer.

Strict Output Format:
<think>Analyze the original question and sub-questions with its answers. Break down the reasoning process step by step. Do NOT provide the final answer yet.</think>
<answer>Provide the final answer based solely on the information before.</answer>

Original Question: {question}
{image_context}
Reasoning Trajectories: {current_reasoning_trajectories}

Based on the information above, give me the final answer to the original question."""

    def _initialize_model(self):
        """初始化模型"""
        logger.info(f"Loading model: {self.model_name}")

        if "VL" in self.model_name:
            # 视觉语言模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        else:
            # 纯语言模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def synthesize_webqa_data(self, webqa_file: str, output_file: str, max_samples: int = 1000):
        """合成WebQA数据的完整推理轨迹"""
        logger.info(f"Synthesizing WebQA data from {webqa_file}")

        with open(webqa_file, 'r', encoding='utf-8') as f:
            webqa_data = [json.loads(line) for line in f]

        synthesized_data = []
        successful_count = 0

        for i, item in enumerate(webqa_data[:max_samples]):
            if i % 10 == 0:
                logger.info(f"Processing WebQA item {i}/{min(max_samples, len(webqa_data))}")

            try:
                trajectory = self._generate_complete_trajectory(item, "webqa")
                if trajectory and self._validate_trajectory(trajectory, item):
                    synthesized_data.append(trajectory)
                    successful_count += 1
                    logger.debug(f"✅ Successfully generated trajectory {successful_count}")
                else:
                    logger.debug(f"❌ Failed to generate valid trajectory for item {i}")
            except Exception as e:
                logger.warning(f"Error processing WebQA item {i}: {e}")
                continue

        # 保存合成数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in synthesized_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"WebQA synthesis completed. Generated {len(synthesized_data)}/{max_samples} valid trajectories")
        return synthesized_data

    def synthesize_mmqa_data(self, mmqa_file: str, output_file: str, max_samples: int = 1000):
        """合成MMQA数据的完整推理轨迹"""
        logger.info(f"Synthesizing MMQA data from {mmqa_file}")

        with open(mmqa_file, 'r', encoding='utf-8') as f:
            mmqa_data = [json.loads(line) for line in f]

        synthesized_data = []
        successful_count = 0

        for i, item in enumerate(mmqa_data[:max_samples]):
            if i % 10 == 0:
                logger.info(f"Processing MMQA item {i}/{min(max_samples, len(mmqa_data))}")

            try:
                trajectory = self._generate_complete_trajectory(item, "mmqa")
                if trajectory and self._validate_trajectory(trajectory, item):
                    synthesized_data.append(trajectory)
                    successful_count += 1
                    logger.debug(f"✅ Successfully generated trajectory {successful_count}")
                else:
                    logger.debug(f"❌ Failed to generate valid trajectory for item {i}")
            except Exception as e:
                logger.warning(f"Error processing MMQA item {i}: {e}")
                continue

        # 保存合成数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in synthesized_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"MMQA synthesis completed. Generated {len(synthesized_data)}/{max_samples} valid trajectories")
        return synthesized_data

    def _generate_complete_trajectory(self, item: Dict[str, Any], dataset: str) -> Optional[Dict[str, Any]]:
        """生成完整的推理和检索轨迹"""

        # 提取问题、答案和图像信息
        question, answer, image_path = self._extract_item_info(item, dataset)
        if not question or not answer:
            return None

        # 处理图像（如果存在）
        image_context = ""
        image_data = None
        if image_path:
            try:
                image_data = self._load_image(image_path)
                image_context = "Note: This question includes visual content that should be considered in your reasoning."
            except Exception as e:
                logger.debug(f"Failed to load image {image_path}: {e}")

        # R1-Router式的完整迭代生成
        reasoning_steps = []
        current_trajectories = ""

        for step in range(self.max_reasoning_steps):
            logger.debug(f"Generating reasoning step {step + 1}")

            # 生成推理步骤（思考 + 子问题 + 检索器选择）
            step_result = self._generate_reasoning_step(
                question, current_trajectories, image_context, image_data, step
            )

            if not step_result:
                logger.debug(f"Failed to generate step {step + 1}, stopping")
                break

            thinking, sub_question, retriever = step_result

            # 如果子问题是"None"，停止迭代
            if sub_question.lower().strip() == "none":
                logger.debug("Sub-question is 'None', stopping iteration")
                break

            # 执行检索操作
            retrieved_info = self._execute_retrieval(
                sub_question, retriever, item, dataset, image_context, image_data
            )

            # 生成中间答案
            intermediate_answer = self._generate_intermediate_answer(
                sub_question, retrieved_info, image_context, image_data
            )

            # 构建推理步骤
            reasoning_step = {
                "step_id": step + 1,
                "thinking": thinking,
                "sub_question": sub_question,
                "selected_retriever": retriever,
                "ground_truth_retriever": retriever,
                "retrieved_info": retrieved_info,
                "intermediate_answer": intermediate_answer
            }

            reasoning_steps.append(reasoning_step)

            # 更新轨迹历史
            current_trajectories += f"Step {step + 1}:\n"
            current_trajectories += f"Thinking: {thinking}\n"
            current_trajectories += f"Sub-question: {sub_question}\n"
            current_trajectories += f"Retriever: {retriever}\n"
            current_trajectories += f"Retrieved: {retrieved_info[:200]}...\n"
            current_trajectories += f"Answer: {intermediate_answer}\n\n"

        # 生成最终答案
        final_thinking, final_answer = self._generate_final_answer(
            question, current_trajectories, image_context, image_data
        )

        # 拒绝采样：检查最终答案是否匹配
        if not self._check_answer_similarity(final_answer, answer):
            logger.debug(f"Answer mismatch: generated='{final_answer[:50]}...', expected='{answer[:50]}...'")
            return None

        # 构建完整轨迹
        trajectory = {
            "question": question,
            "final_answer": answer,
            "reasoning_steps": reasoning_steps,
            "final_thinking": final_thinking,
            "dataset": dataset,
            "metadata": self._extract_metadata(item, dataset),
            "has_image": image_path is not None,
            "image_path": image_path if image_path else None
        }

        return trajectory

    def _extract_item_info(self, item: Dict[str, Any], dataset: str) -> Tuple[str, str, Optional[str]]:
        """提取项目的基本信息"""
        if dataset == "webqa":
            question = item.get('Q', '')
            answer = item.get('A', '')
            # WebQA中图像信息的处理
            image_path = None
            img_facts = item.get('img_posFacts', [])
            if img_facts:
                # 可以根据实际数据结构调整
                image_path = f"webqa_image_{item.get('Guid', 'unknown')}"

        elif dataset == "mmqa":
            question = item.get('question', '')
            answers = item.get('answers', [])
            answer = answers[0].get('answer', '') if answers else ''
            # MMQA中图像信息的处理
            image_path = None
            modalities = item.get('modalities', [])
            if 'image' in modalities:
                # 根据实际数据结构调整
                image_instances = item.get('image_instances', [])
                if image_instances:
                    image_path = f"mmqa_image_{item.get('qid', 'unknown')}"
        else:
            return "", "", None

        return question, answer, image_path

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图像（如果路径存在）"""
        # 这里可以根据实际的图像存储位置进行调整
        # 目前返回None，因为我们主要关注推理轨迹生成
        return None

    def _generate_reasoning_step(self, question: str, current_trajectories: str,
                                 image_context: str, image_data: Optional[Image.Image],
                                 step: int) -> Optional[Tuple[str, str, str]]:
        """生成单个推理步骤"""
        try:
            # 构建提示词
            prompt = self.system_prompt.format(
                question=question,
                image_context=image_context,
                current_reasoning_trajectories=current_trajectories
            )

            # 调用模型生成
            response = self._call_model(prompt, image_data)

            # 解析响应
            thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            subq_match = re.search(r'<sub-question>(.*?)</sub-question>', response, re.DOTALL)
            ret_match = re.search(r'<ret>(.*?)</ret>', response, re.DOTALL)

            if not all([thinking_match, subq_match, ret_match]):
                logger.debug(f"Failed to parse step response: {response}")
                return None

            thinking = thinking_match.group(1).strip()
            sub_question = subq_match.group(1).strip()
            retriever = ret_match.group(1).strip()

            return thinking, sub_question, retriever

        except Exception as e:
            logger.debug(f"Error in generating reasoning step: {e}")
            return None

    def _execute_retrieval(self, sub_question: str, retriever: str, item: Dict,
                           dataset: str, image_context: str,
                           image_data: Optional[Image.Image]) -> str:
        """执行检索操作（整合版本）"""
        return self._real_retrieval(sub_question, retriever, item, dataset)

    def _real_retrieval(self, sub_question: str, retriever: str, item: Dict, dataset: str) -> str:
        """真实检索实现（可扩展）"""
        # 这里可以实现真实的检索逻辑
        # 包括文本检索、图像检索、表格检索等

        if retriever == "Text Retriever":
            # 实现文本检索逻辑
            pass
        elif retriever == "Text Image Retriever":
            # 实现图文检索逻辑
            pass
        elif retriever == "Table Retriever":
            # 实现表格检索逻辑
            pass

        # 目前返回模拟结果
        return self._simulate_retrieval(sub_question, retriever, item, dataset)

    def _generate_intermediate_answer(self, sub_question: str, retrieved_info: str,
                                      image_context: str, image_data: Optional[Image.Image]) -> str:
        """生成中间答案"""
        try:
            prompt = self.intermediate_answer_prompt.format(
                sub_question=sub_question,
                retrieved_info=retrieved_info,
                image_context=image_context
            )

            response = self._call_model(prompt, image_data)

            # 解析答案
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                return answer_match.group(1).strip()
            else:
                # 如果没有找到标签，返回整个响应
                return response.strip()

        except Exception as e:
            logger.debug(f"Error generating intermediate answer: {e}")
            return f"Based on the retrieved information, I can provide relevant details about: {sub_question}"

    def _generate_final_answer(self, question: str, current_trajectories: str,
                               image_context: str, image_data: Optional[Image.Image]) -> Tuple[str, str]:
        """生成最终答案"""
        try:
            prompt = self.final_answer_prompt.format(
                question=question,
                image_context=image_context,
                current_reasoning_trajectories=current_trajectories
            )

            response = self._call_model(prompt, image_data)

            # 解析思考和答案
            thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

            thinking = thinking_match.group(
                1).strip() if thinking_match else "Based on the previous steps, I can now provide the final answer."
            answer = answer_match.group(1).strip() if answer_match else response.strip()

            return thinking, answer

        except Exception as e:
            logger.debug(f"Error generating final answer: {e}")
            return "Summarizing the information gathered from all steps.", "Unable to generate final answer."

    def _call_model(self, prompt: str, image_data: Optional[Image.Image] = None, max_length: int = 1024) -> str:
        """调用模型生成响应（支持图像输入）"""
        try:
            if self.processor:  # 视觉语言模型
                content = [{"type": "text", "text": prompt}]

                # 如果有图像数据，添加到内容中
                if image_data:
                    content.insert(0, {"type": "image", "image": image_data})

                messages = [{"role": "user", "content": content}]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.processor(
                    text=[text],
                    images=[image_data] if image_data else None,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            else:  # 纯语言模型
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    generated_ids[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True
                )

            return response.strip()

        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return ""

    def _check_answer_similarity(self, generated: str, expected: str) -> bool:
        """检查答案相似性（简化的拒绝采样）"""
        # 简单的相似性检查
        generated_clean = generated.lower().strip()
        expected_clean = expected.lower().strip()

        # 如果答案较短，使用完全匹配
        if len(expected_clean) < 10:
            return generated_clean == expected_clean

        # 对于较长答案，检查关键词覆盖
        expected_words = set(expected_clean.split())
        generated_words = set(generated_clean.split())

        if len(expected_words) == 0:
            return True

        overlap = len(expected_words.intersection(generated_words))
        coverage = overlap / len(expected_words)

        return coverage >= 0.3  # 30%的关键词覆盖率

    def _validate_trajectory(self, trajectory: Dict, item: Dict) -> bool:
        """验证生成的轨迹质量"""
        # 基本字段检查
        required_fields = ['question', 'final_answer', 'reasoning_steps']
        for field in required_fields:
            if field not in trajectory:
                return False

        # 检查推理步骤
        if not trajectory['reasoning_steps']:
            # 允许没有推理步骤的简单问题
            return True

        # 检查每个步骤的完整性
        for step in trajectory['reasoning_steps']:
            required_step_fields = ['thinking', 'sub_question', 'selected_retriever', 'intermediate_answer']
            for field in required_step_fields:
                if field not in step or not step[field]:
                    return False

        return True

    def _extract_metadata(self, item: Dict, dataset: str) -> Dict:
        """提取元数据"""
        if dataset == "webqa":
            return {
                "guid": item.get('Guid', ''),
                "has_image": len(item.get('img_posFacts', [])) > 0,
                "has_text": len(item.get('txt_posFacts', [])) > 0,
                "num_images": len(item.get('img_posFacts', [])),
                "num_text_facts": len(item.get('txt_posFacts', [])),
                "image_ids": item.get('img_posFacts', [])
            }
        elif dataset == "mmqa":
            return {
                "qid": item.get('qid', ''),
                "modalities": item.get('modalities', []),
                "dataset_type": item.get('type', 'unknown'),
                "answer_type": item.get('answers', [{}])[0].get('type', 'string') if item.get('answers') else 'string',
                "image_instances": item.get('image_instances', [])
            }
        return {}

    def enable_retrieval_mode(self, enable: bool = True):
        """启用/禁用真实检索模式"""
        self.enable_real_retrieval = enable
        if enable:
            logger.info("Real retrieval mode enabled")
        else:
            logger.info("Using high-quality simulation mode")


def main():
    parser = argparse.ArgumentParser(description="Integrated R1-Router data synthesis for WebQA and MMQA")
    parser.add_argument("--webqa_file", type=str, help="Path to WebQA data file")
    parser.add_argument("--mmqa_file", type=str, help="Path to MMQA data file")
    parser.add_argument("--output_dir", type=str, default="./integrated_r1_data", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to process per dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name for generation")
    parser.add_argument("--enable_real_retrieval", action="store_true", help="Enable real retrieval (requires setup)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging level")

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化合成器
    logger.info(f"Initializing Integrated R1-Router synthesizer with model: {args.model_name}")
    synthesizer = IntegratedR1RouterSynthesizer(args.model_name)

    # 设置检索模式
    synthesizer.enable_retrieval_mode(args.enable_real_retrieval)

    # 处理WebQA数据
    if args.webqa_file and Path(args.webqa_file).exists():
        webqa_output = output_dir / "webqa_integrated_r1.jsonl"
        logger.info(f"Processing WebQA file: {args.webqa_file}")
        synthesizer.synthesize_webqa_data(
            args.webqa_file,
            str(webqa_output),
            args.max_samples
        )
        logger.info(f"WebQA results saved to: {webqa_output}")

    # 处理MMQA数据
    if args.mmqa_file and Path(args.mmqa_file).exists():
        mmqa_output = output_dir / "mmqa_integrated_r1.jsonl"
        logger.info(f"Processing MMQA file: {args.mmqa_file}")
        synthesizer.synthesize_mmqa_data(
            args.mmqa_file,
            str(mmqa_output),
            args.max_samples
        )
        logger.info(f"MMQA results saved to: {mmqa_output}")

    # 清理内存
    if hasattr(synthesizer, 'model'):
        del synthesizer.model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Integrated R1-Router data synthesis completed!")


if __name__ == "__main__":
    main()