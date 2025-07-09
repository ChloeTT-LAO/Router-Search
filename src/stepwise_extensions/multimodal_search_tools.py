import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MultimodalKnowledgeBase:
    """多模态知识库，支持文本、图像、表格检索"""

    def __init__(self, kb_config: Dict[str, str]):
        self.kb_config = kb_config
        self.text_encoder = SentenceTransformer('BAAI/bge-m3')
        self.image_encoder = None  # 将在需要时加载

        # 加载预构建的知识库索引
        self.text_index = self._load_faiss_index(kb_config['text_index_path'])
        self.image_index = self._load_faiss_index(kb_config['image_index_path'])
        self.table_index = self._load_faiss_index(kb_config['table_index_path'])

        # 加载对应的文档
        self.text_docs = self._load_documents(kb_config['text_docs_path'])
        self.image_docs = self._load_documents(kb_config['image_docs_path'])
        self.table_docs = self._load_documents(kb_config['table_docs_path'])

    def _load_faiss_index(self, index_path: str) -> faiss.Index:
        """加载FAISS索引"""
        if not Path(index_path).exists():
            logger.warning(f"Index file not found: {index_path}")
            return None
        return faiss.read_index(index_path)

    def _load_documents(self, docs_path: str) -> List[Dict]:
        """加载文档数据"""
        if not Path(docs_path).exists():
            logger.warning(f"Documents file not found: {docs_path}")
            return []

        with open(docs_path, 'rb') as f:
            return pickle.load(f)


class MultimodalSearchTools:
    """多模态搜索工具，集成StepSearch的信息增益和冗余惩罚机制"""

    def __init__(self, kb_config: Dict[str, str]):
        self.knowledge_base = MultimodalKnowledgeBase(kb_config)
        self.search_history = []
        self.sentence_encoder = SentenceTransformer('BAAI/bge-m3')

    def text_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """在文本知识库中搜索"""
        if self.knowledge_base.text_index is None:
            return {"search_type": "text", "query": query, "results": [], "scores": []}

        # 编码查询
        query_embedding = self.sentence_encoder.encode([query])

        # 在FAISS索引中搜索
        scores, indices = self.knowledge_base.text_index.search(query_embedding, top_k)

        # 获取搜索结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base.text_docs):
                doc = self.knowledge_base.text_docs[idx]
                results.append({
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "score": float(score),
                    "source": "text_kb"
                })

        return {
            "search_type": "text",
            "query": query,
            "results": results,
            "scores": scores[0].tolist()
        }

    def image_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """在图像知识库中搜索（基于图像描述）"""
        if self.knowledge_base.image_index is None:
            return {"search_type": "image", "query": query, "results": [], "scores": []}

        # 编码查询
        query_embedding = self.sentence_encoder.encode([query])

        # 在FAISS索引中搜索
        scores, indices = self.knowledge_base.image_index.search(query_embedding, top_k)

        # 获取搜索结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base.image_docs):
                doc = self.knowledge_base.image_docs[idx]
                results.append({
                    "content": doc.get("caption", ""),
                    "image_path": doc.get("image_path", ""),
                    "title": doc.get("title", ""),
                    "score": float(score),
                    "source": "image_kb"
                })

        return {
            "search_type": "image",
            "query": query,
            "results": results,
            "scores": scores[0].tolist()
        }

    def table_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """在表格知识库中搜索"""
        if self.knowledge_base.table_index is None:
            return {"search_type": "table", "query": query, "results": [], "scores": []}

        # 编码查询
        query_embedding = self.sentence_encoder.encode([query])

        # 在FAISS索引中搜索
        scores, indices = self.knowledge_base.table_index.search(query_embedding, top_k)

        # 获取搜索结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.knowledge_base.table_docs):
                doc = self.knowledge_base.table_docs[idx]
                results.append({
                    "content": doc.get("table_content", ""),
                    "title": doc.get("title", ""),
                    "headers": doc.get("headers", []),
                    "rows": doc.get("rows", []),
                    "score": float(score),
                    "source": "table_kb"
                })

        return {
            "search_type": "table",
            "query": query,
            "results": results,
            "scores": scores[0].tolist()
        }

    def calculate_information_gain(self, new_results: Dict[str, Any],
                                   existing_info: List[Dict[str, Any]]) -> float:
        """计算信息增益 - StepSearch核心功能"""
        if not new_results.get('../results'):
            return 0.0

        # 提取新信息内容
        new_content = []
        for result in new_results['results']:
            new_content.append(result.get('content', ''))
        new_text = ' '.join(new_content)

        if not existing_info or not new_text.strip():
            return 1.0  # 完全新信息

        # 提取历史信息内容
        existing_content = []
        for info in existing_info:
            if 'search_results' in info:
                for result in info['search_results'].get('results', []):
                    existing_content.append(result.get('content', ''))
        existing_text = ' '.join(existing_content)

        if not existing_text.strip():
            return 1.0

        # 计算语义相似度
        embeddings = self.sentence_encoder.encode([new_text, existing_text])
        similarity = F.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item()

        # 信息增益 = 1 - 相似度
        information_gain = max(0.0, 1.0 - similarity)

        return information_gain

    def calculate_redundancy_penalty(self, current_query: str,
                                     search_history: List[Dict[str, Any]]) -> float:
        """计算冗余惩罚 - StepSearch核心功能"""
        if not search_history or not current_query.strip():
            return 0.0

        # 提取历史查询
        historical_queries = []
        for entry in search_history:
            if 'query' in entry and entry['query'].strip():
                historical_queries.append(entry['query'])

        if not historical_queries:
            return 0.0

        # 计算与历史查询的最大相似度
        all_queries = [current_query] + historical_queries
        embeddings = self.sentence_encoder.encode(all_queries)

        current_embedding = torch.tensor(embeddings[0]).unsqueeze(0)
        historical_embeddings = torch.tensor(embeddings[1:])

        similarities = F.cosine_similarity(
            current_embedding,
            historical_embeddings,
            dim=1
        )

        max_similarity = similarities.max().item()

        return max_similarity

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self.sentence_encoder.encode([text1, text2])
        similarity = F.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item()

        return similarity

    def route_query(self, query: str, context: str = "",
                    available_retrievers: List[str] = None) -> str:
        """智能路由查询到最合适的检索器"""
        if available_retrievers is None:
            available_retrievers = ["Text Retriever", "Text Image Retriever", "Table Retriever"]

        # 简单的启发式路由规则
        query_lower = query.lower()
        context_lower = context.lower()

        # 图像相关关键词
        image_keywords = [
            'image', 'picture', 'photo', 'visual', 'see', 'show', 'look',
            'appears', 'depicts', 'illustration', 'figure', 'diagram'
        ]

        # 表格相关关键词
        table_keywords = [
            'table', 'data', 'statistics', 'numbers', 'percentage', 'rate',
            'comparison', 'list', 'rank', 'top', 'count', 'total', 'average'
        ]

        # 检查是否需要图像检索
        if any(keyword in query_lower or keyword in context_lower for keyword in image_keywords):
            if "Text Image Retriever" in available_retrievers:
                return "Text Image Retriever"

        # 检查是否需要表格检索
        if any(keyword in query_lower or keyword in context_lower for keyword in table_keywords):
            if "Table Retriever" in available_retrievers:
                return "Table Retriever"

        # 默认使用文本检索
        return "Text Retriever"

    def execute_search(self, query: str, retriever_type: str, top_k: int = 5) -> Dict[str, Any]:
        """执行搜索操作"""
        if retriever_type == "Text Retriever":
            return self.text_search(query, top_k)
        elif retriever_type == "Text Image Retriever":
            return self.image_search(query, top_k)
        elif retriever_type == "Table Retriever":
            return self.table_search(query, top_k)
        else:
            # 默认文本搜索
            return self.text_search(query, top_k)