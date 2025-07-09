#!/usr/bin/env python3
"""
知识库构建脚本
从原始数据构建FAISS索引，支持文本、图像、表格三种模态
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self, output_dir: str = "./knowledge_bases"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "image").mkdir(exist_ok=True)
        (self.output_dir / "table").mkdir(exist_ok=True)

        # 初始化编码器
        self.text_encoder = SentenceTransformer('BAAI/bge-m3')

    def build_text_knowledge_base(self,
                                  text_corpus_file: str,
                                  chunk_size: int = 512,
                                  batch_size: int = 32):
        """构建文本知识库"""
        logger.info(f"Building text knowledge base from {text_corpus_file}")

        # 加载文本语料
        text_docs = []
        with open(text_corpus_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())

                    # 处理文本分块
                    content = doc.get('content', doc.get('text', ''))
                    title = doc.get('title', f"Document_{line_idx}")

                    if content.strip():
                        # 简单分块策略
                        chunks = self._split_text(content, chunk_size)

                        for chunk_idx, chunk in enumerate(chunks):
                            text_docs.append({
                                'content': chunk,
                                'title': title,
                                'doc_id': f"{line_idx}_{chunk_idx}",
                                'source': doc.get('source', 'unknown')
                            })

                    if line_idx % 10000 == 0:
                        logger.info(f"Processed {line_idx} documents, {len(text_docs)} chunks")

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_idx}")
                    continue

        logger.info(f"Total text chunks: {len(text_docs)}")

        # 构建嵌入
        logger.info("Encoding text chunks...")
        embeddings = []

        for i in tqdm(range(0, len(text_docs), batch_size), desc="Encoding"):
            batch_texts = [doc['content'] for doc in text_docs[i:i + batch_size]]
            batch_embeddings = self.text_encoder.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings).astype('float32')

        # 构建FAISS索引
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 内积索引，适合归一化嵌入
        index.add(embeddings)

        # 保存索引和文档
        index_path = self.output_dir / "text" / "text_index.faiss"
        docs_path = self.output_dir / "text" / "text_docs.pkl"

        faiss.write_index(index, str(index_path))

        with open(docs_path, 'wb') as f:
            pickle.dump(text_docs, f)

        logger.info(f"Text knowledge base saved:")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Documents: {docs_path}")
        logger.info(f"  Total documents: {len(text_docs)}")

        return index_path, docs_path

    def build_image_knowledge_base(self,
                                   image_corpus_file: str,
                                   batch_size: int = 32):
        """构建图像知识库（基于图像描述）"""
        logger.info(f"Building image knowledge base from {image_corpus_file}")

        # 加载图像语料
        image_docs = []
        with open(image_corpus_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())

                    caption = doc.get('caption', doc.get('description', ''))
                    image_path = doc.get('image_path', doc.get('path', ''))
                    title = doc.get('title', f"Image_{line_idx}")

                    if caption.strip():
                        image_docs.append({
                            'caption': caption,
                            'image_path': image_path,
                            'title': title,
                            'doc_id': f"img_{line_idx}",
                            'source': doc.get('source', 'unknown')
                        })

                    if line_idx % 5000 == 0:
                        logger.info(f"Processed {line_idx} images, {len(image_docs)} valid")

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_idx}")
                    continue

        logger.info(f"Total image documents: {len(image_docs)}")

        # 构建嵌入（基于图像描述）
        logger.info("Encoding image captions...")
        embeddings = []

        for i in tqdm(range(0, len(image_docs), batch_size), desc="Encoding"):
            batch_captions = [doc['caption'] for doc in image_docs[i:i + batch_size]]
            batch_embeddings = self.text_encoder.encode(
                batch_captions,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings).astype('float32')

        # 构建FAISS索引
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # 保存索引和文档
        index_path = self.output_dir / "image" / "image_index.faiss"
        docs_path = self.output_dir / "image" / "image_docs.pkl"

        faiss.write_index(index, str(index_path))

        with open(docs_path, 'wb') as f:
            pickle.dump(image_docs, f)

        logger.info(f"Image knowledge base saved:")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Documents: {docs_path}")
        logger.info(f"  Total documents: {len(image_docs)}")

        return index_path, docs_path

    def build_table_knowledge_base(self,
                                   table_corpus_file: str,
                                   batch_size: int = 32):
        """构建表格知识库"""
        logger.info(f"Building table knowledge base from {table_corpus_file}")

        # 加载表格语料
        table_docs = []
        with open(table_corpus_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())

                    title = doc.get('title', f"Table_{line_idx}")
                    headers = doc.get('headers', [])
                    rows = doc.get('rows', [])

                    # 构建表格文本表示
                    table_content = self._table_to_text(title, headers, rows)

                    if table_content.strip():
                        table_docs.append({
                            'table_content': table_content,
                            'title': title,
                            'headers': headers,
                            'rows': rows,
                            'doc_id': f"table_{line_idx}",
                            'source': doc.get('source', 'unknown')
                        })

                    if line_idx % 2000 == 0:
                        logger.info(f"Processed {line_idx} tables, {len(table_docs)} valid")

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_idx}")
                    continue

        logger.info(f"Total table documents: {len(table_docs)}")

        # 构建嵌入
        logger.info("Encoding table contents...")
        embeddings = []

        for i in tqdm(range(0, len(table_docs), batch_size), desc="Encoding"):
            batch_contents = [doc['table_content'] for doc in table_docs[i:i + batch_size]]
            batch_embeddings = self.text_encoder.encode(
                batch_contents,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings).astype('float32')

        # 构建FAISS索引
        logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # 保存索引和文档
        index_path = self.output_dir / "table" / "table_index.faiss"
        docs_path = self.output_dir / "table" / "table_docs.pkl"

        faiss.write_index(index, str(index_path))

        with open(docs_path, 'wb') as f:
            pickle.dump(table_docs, f)

        logger.info(f"Table knowledge base saved:")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Documents: {docs_path}")
        logger.info(f"  Total documents: {len(table_docs)}")

        return index_path, docs_path

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """分割文本为块"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _table_to_text(self, title: str, headers: List[str], rows: List[List[str]]) -> str:
        """将表格转换为文本表示"""
        text_parts = []

        if title:
            text_parts.append(f"Table: {title}")

        if headers:
            text_parts.append(f"Headers: {' | '.join(headers)}")

        if rows:
            text_parts.append("Rows:")
            for i, row in enumerate(rows[:10]):  # 限制行数
                if len(row) == len(headers):
                    row_text = ' | '.join(str(cell) for cell in row)
                    text_parts.append(f"Row {i + 1}: {row_text}")
                else:
                    row_text = ' | '.join(str(cell) for cell in row)
                    text_parts.append(f"Row {i + 1}: {row_text}")

        return '\n'.join(text_parts)

    def build_all_knowledge_bases(self,
                                  text_corpus: str = None,
                                  image_corpus: str = None,
                                  table_corpus: str = None):
        """构建所有知识库"""
        logger.info("Starting knowledge base construction...")

        kb_paths = {}

        # 构建文本知识库
        if text_corpus and os.path.exists(text_corpus):
            text_index, text_docs = self.build_text_knowledge_base(text_corpus)
            kb_paths['text'] = {'index': str(text_index), 'docs': str(text_docs)}

        # 构建图像知识库
        if image_corpus and os.path.exists(image_corpus):
            image_index, image_docs = self.build_image_knowledge_base(image_corpus)
            kb_paths['image'] = {'index': str(image_index), 'docs': str(image_docs)}

        # 构建表格知识库
        if table_corpus and os.path.exists(table_corpus):
            table_index, table_docs = self.build_table_knowledge_base(table_corpus)
            kb_paths['table'] = {'index': str(table_index), 'docs': str(table_docs)}

        # 生成配置文件
        config = {
            "text_index_path": kb_paths.get('text', {}).get('index', ''),
            "text_docs_path": kb_paths.get('text', {}).get('docs', ''),
            "image_index_path": kb_paths.get('image', {}).get('index', ''),
            "image_docs_path": kb_paths.get('image', {}).get('docs', ''),
            "table_index_path": kb_paths.get('table', {}).get('index', ''),
            "table_docs_path": kb_paths.get('table', {}).get('docs', ''),
            "retriever_configs": {
                "text_retriever": {
                    "model_name": "BAAI/bge-m3",
                    "max_length": 512,
                    "normalize_embeddings": True
                },
                "image_retriever": {
                    "model_name": "BAAI/bge-m3",
                    "max_length": 512,
                    "normalize_embeddings": True
                },
                "table_retriever": {
                    "model_name": "BAAI/bge-m3",
                    "max_length": 512,
                    "normalize_embeddings": True
                }
            },
            "search_settings": {
                "default_top_k": 5,
                "max_search_rounds": 5,
                "information_gain_threshold": 0.1,
                "redundancy_penalty_threshold": 0.7
            }
        }

        config_path = self.output_dir / "kb_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Knowledge base configuration saved to: {config_path}")
        logger.info("Knowledge base construction completed!")

        return config_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Build knowledge bases for multimodal RAG")
    parser.add_argument("--text_corpus", type=str, help="Path to text corpus file (JSONL)")
    parser.add_argument("--image_corpus", type=str, help="Path to image corpus file (JSONL)")
    parser.add_argument("--table_corpus", type=str, help="Path to table corpus file (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./knowledge_bases",
                        help="Output directory for knowledge bases")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Text chunk size for splitting")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")

    args = parser.parse_args()

    # 检查输入
    if not any([args.text_corpus, args.image_corpus, args.table_corpus]):
        logger.error("At least one corpus file must be provided")
        return

    # 初始化构建器
    builder = KnowledgeBaseBuilder(args.output_dir)

    # 构建知识库
    config_path = builder.build_all_knowledge_bases(
        text_corpus=args.text_corpus,
        image_corpus=args.image_corpus,
        table_corpus=args.table_corpus
    )

    logger.info(f"Knowledge bases built successfully!")
    logger.info(f"Configuration file: {config_path}")


if __name__ == "__main__":
    main()