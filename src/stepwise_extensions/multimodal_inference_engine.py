import json
import re
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MultimodalInferenceEngine:
    """
    多模态推理引擎，融合StepSearch的逐步推理和R1-Router的知识库路由
    """

    def __init__(self,
                 model_path: str,
                 kb_config: Dict[str, str],
                 max_search_rounds: int = 5,
                 temperature: float = 0.1):

        self.model_path = model_path
        self.max_search_rounds = max_search_rounds
        self.temperature = temperature

        # 加载模型和处理器
        self.model = None
        self.processor = None
        self._load_model()

        # 初始化搜索工具
        from src.stepwise_extensions.multimodal_search_tools import MultimodalSearchTools
        self.search_tools = MultimodalSearchTools(kb_config)

        # 推理模板
        self.prompt_template = self._get_prompt_template()

    def _load_model(self):
        """加载模型"""
        logger.info(f"Loading model from {self.model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate_response(self,
                          question: str,
                          image: Optional[Image.Image] = None,
                          context: str = "") -> Dict[str, Any]:
        """
        生成多模态推理响应

        Args:
            question: 用户问题
            image: 输入图像（可选）
            context: 额外上下文

        Returns:
            包含答案、推理轨迹、搜索历史的字典
        """

        search_history = []
        reasoning_trajectory = []
        current_context = context

        logger.info(f"Starting inference for question: {question}")

        for round_idx in range(self.max_search_rounds):
            logger.debug(f"Reasoning round {round_idx + 1}")

            # 第一步：生成推理和决定下一步动作
            reasoning_result = self._generate_reasoning_step(
                question=question,
                image=image,
                current_context=current_context,
                search_history=search_history,
                round_idx=round_idx
            )

            reasoning_trajectory.append(reasoning_result)

            # 第二步：根据推理结果决定是否需要搜索
            if reasoning_result.get('needs_search', False):
                # 执行搜索
                search_result = self._execute_search_step(
                    reasoning_result=reasoning_result,
                    search_history=search_history
                )

                if search_result:
                    search_history.append(search_result)

                    # 更新上下文
                    current_context = self._update_context(
                        current_context,
                        search_result
                    )
            else:
                # 生成最终答案
                final_answer = reasoning_result.get('final_answer', '')
                logger.info(f"Generated final answer: {final_answer}")
                break
        else:
            # 如果达到最大轮数，生成最终答案
            final_answer = self._generate_final_answer(
                question, image, current_context, reasoning_trajectory
            )

        return {
            'final_answer': final_answer,
            'reasoning_trajectory': reasoning_trajectory,
            'search_history': search_history,
            'total_search_rounds': len(search_history)
        }

    def _generate_reasoning_step(self,
                                 question: str,
                                 image: Optional[Image.Image],
                                 current_context: str,
                                 search_history: List[Dict],
                                 round_idx: int) -> Dict[str, Any]:
        """生成单步推理"""

        # 构建提示
        prompt = self._build_reasoning_prompt(
            question, current_context, search_history, round_idx
        )

        # 准备输入
        messages = [{"role": "user", "content": prompt}]

        if image is not None:
            messages[0]["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]

        # 生成响应
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # 解析响应
        parsed_result = self._parse_reasoning_response(response)
        parsed_result['round_idx'] = round_idx
        parsed_result['raw_response'] = response

        return parsed_result

    def _execute_search_step(self,
                             reasoning_result: Dict[str, Any],
                             search_history: List[Dict]) -> Optional[Dict[str, Any]]:
        """执行搜索步骤"""

        query = reasoning_result.get('search_query', '')
        retriever_type = reasoning_result.get('selected_retriever', 'Text Retriever')

        if not query.strip():
            logger.warning("Empty search query, skipping search")
            return None

        logger.info(f"Executing search: '{query}' with {retriever_type}")

        # 执行搜索
        search_results = self.search_tools.execute_search(
            query=query,
            retriever_type=retriever_type,
            top_k=5
        )

        # 计算信息增益和冗余惩罚
        information_gain = self.search_tools.calculate_information_gain(
            search_results, search_history
        )

        redundancy_penalty = self.search_tools.calculate_redundancy_penalty(
            query, search_history
        )

        search_step_result = {
            'query': query,
            'retriever_type': retriever_type,
            'search_results': search_results,
            'information_gain': information_gain,
            'redundancy_penalty': redundancy_penalty,
            'step_quality': information_gain - redundancy_penalty
        }

        logger.debug(f"Search completed. Info gain: {information_gain:.3f}, "
                     f"Redundancy: {redundancy_penalty:.3f}")

        return search_step_result

    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """解析推理响应"""
        result = {
            'needs_search': False,
            'search_query': '',
            'selected_retriever': '',
            'thinking': '',
            'final_answer': ''
        }

        # 提取thinking内容
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response, re.DOTALL)
        if think_match:
            result['thinking'] = think_match.group(1).strip()

        # 提取sub-question
        subq_pattern = r'<sub-question>(.*?)</sub-question>'
        subq_match = re.search(subq_pattern, response, re.DOTALL)
        if subq_match:
            result['search_query'] = subq_match.group(1).strip()
            result['needs_search'] = True

        # 提取retriever选择
        ret_pattern = r'<ret>(.*?)</ret>'
        ret_match = re.search(ret_pattern, response, re.DOTALL)
        if ret_match:
            result['selected_retriever'] = ret_match.group(1).strip()

        # 提取最终答案
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        if answer_match:
            result['final_answer'] = answer_match.group(1).strip()
            result['needs_search'] = False

        # 如果没有明确的搜索指令，但也没有最终答案，则生成搜索查询
        if not result['needs_search'] and not result['final_answer']:
            # 智能路由决定检索器类型
            available_retrievers = ["Text Retriever", "Text Image Retriever", "Table Retriever"]
            suggested_retriever = self.search_tools.route_query(
                result['thinking'],
                "",
                available_retrievers
            )

            result['needs_search'] = True
            result['search_query'] = result['thinking'][:100] if result['thinking'] else "relevant information"
            result['selected_retriever'] = suggested_retriever

        return result

    def _update_context(self,
                        current_context: str,
                        search_result: Dict[str, Any]) -> str:
        """更新上下文"""

        search_info = []

        # 添加搜索结果到上下文
        if 'search_results' in search_result and search_result['search_results'].get('results'):
            for i, result in enumerate(search_result['search_results']['results'][:3]):
                content = result.get('content', '')
                if content.strip():
                    search_info.append(f"Search result {i + 1}: {content[:200]}...")

        if search_info:
            new_context = current_context + "\n\nRetrieved information:\n" + "\n".join(search_info)
        else:
            new_context = current_context

        return new_context

    def _generate_final_answer(self,
                               question: str,
                               image: Optional[Image.Image],
                               context: str,
                               reasoning_trajectory: List[Dict]) -> str:
        """生成最终答案"""

        # 构建最终答案提示
        prompt = f"""Based on the question and all the information gathered through reasoning and search, provide a comprehensive final answer.

Question: {question}

Context and Retrieved Information:
{context}

Reasoning Trajectory:
"""

        for i, step in enumerate(reasoning_trajectory):
            prompt += f"Step {i + 1}: {step.get('thinking', '')}\n"

        prompt += "\nFinal Answer:"

        # 准备输入
        messages = [{"role": "user", "content": prompt}]

        if image is not None:
            messages[0]["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]

        # 生成响应
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        final_answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return final_answer

    def _build_reasoning_prompt(self,
                                question: str,
                                context: str,
                                search_history: List[Dict],
                                round_idx: int) -> str:
        """构建推理提示"""

        prompt = f"""You are a professional question-answering expert for multi-hop QA systems. Your task is to decompose complex questions into strictly single-hop sub-questions and select appropriate retrievers.

Strict Output Format:
<think>[Analyze the original question and determine the next required sub-question. Do NOT reveal answers or perform multi-hop reasoning. If no further information is needed to answer the original question, write "None".]/</think>
<sub-question>[Exactly ONE single-hop question one line. If no further information is needed to answer the original question, write "None".]/</sub-question><ret>[Choose 1 retriever from: Text Retriever, Text Image Retriever, Table Retriever. Write "None" if <sub-question> is "None".]/</ret>

Critical Rules:
1. Atomic Sub-question Definition:
   a) A sub-question is "atomic" only if:
   b) It cannot be further decomposed into simpler questions
   c) It requires exactly "one retrieval action" to answer
   d) Does NOT depend on answers to previous sub-questions
   e) It can NOT be helpful to answer the origin question
   - Example: ✗ "Find the capital and population of France" → ✓ Split into two sub-questions

2. Retriever Selection Guidelines:
   - "Text Retriever":
     - For non-visual commonsense knowledge (e.g., "Define photosynthesis")
     - For general factual information
   - "Text Image Retriever":
     - When sub-question explicitly references visual elements (e.g., "Describe the painting style of...")
   - "Table Retriever":
     - For numerical/statistical queries (e.g., "GDP of Japan in 2020")

3. Strict Prohibitions:
   - Never combine multiple questions in <sub-question>
   - Never mention retrieved content in <think>
   - Never select retrievers for non-atomic questions

Question: {question}
"""

        if context:
            prompt += f"\nCurrent Context: {context}"

        if search_history:
            prompt += f"\nPrevious Search Steps: {len(search_history)}"
            for i, step in enumerate(search_history[-3:]):  # 只显示最近3步
                prompt += f"\nStep {i + 1}: Query='{step.get('query', '')}', Retriever={step.get('retriever_type', '')}"

        if round_idx > 0:
            prompt += f"\n\nThis is reasoning step {round_idx + 1}. If you have sufficient information to answer the original question, provide the final answer using <answer></answer> tags instead of searching."

        return prompt

    def _get_prompt_template(self) -> str:
        """获取提示模板"""
        return """You are a professional question answering model. Your task is to carefully think through the question based on the information retrieved and then provide the final answer.

Strict Output Format:
<think>
[Analyze the original question and the retrieved information. Break down the reasoning process step by step. Do NOT provide the final answer yet.]
</think>
<answer>
[Provide the final answer based solely on the retrieved information.]
</answer>

According to the related information searched, "ter" means this info is from text retriever, "tir" means this info is from text image retriever, "tar" means this info is from table retriever.(document)\n\n Give me the answer(with the format <answer></answer>) to the Question: {question}"""


def main():
    """测试推理引擎"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--kb_config", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)

    args = parser.parse_args()

    # 加载知识库配置
    with open(args.kb_config, 'r') as f:
        kb_config = json.load(f)

    # 初始化推理引擎
    engine = MultimodalInferenceEngine(args.model_path, kb_config)

    # 加载图像（如果提供）
    image = None
    if args.image_path:
        image = Image.open(args.image_path)

    # 执行推理
    result = engine.generate_response(args.question, image)

    # 输出结果
    print("=" * 50)
    print(f"Question: {args.question}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Total Search Rounds: {result['total_search_rounds']}")
    print("=" * 50)


if __name__ == "__main__":
    main()