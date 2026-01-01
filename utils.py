import os
import numpy as np
import hashlib
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
import torch
from config import DEVICE, TEXT_MODEL, IMAGE_MODEL, PAPER_ROOT,PAPER_COLLECTION, PAPER_TOPICS,CLASSIFICATION_THRESHOLD

def extract_pdf_text(pdf_path):
    """提取PDF文本内容"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text.strip()
    except Exception as e:
        print(f"提取PDF文本失败: {e}")
        return ""

def get_text_embedding(text):
    """生成文本嵌入向量"""
    if not text or len(text) < 10:
        return None
    max_length = 2000
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    embeddings = TEXT_MODEL.encode(chunks, convert_to_tensor=False)
    avg_embedding = embeddings.mean(axis=0).tolist()
    return avg_embedding

def get_image_embedding(image_path):
    """生成图像嵌入向量（512维+归一化，修复维度歧义错误）"""
    try:
        image = Image.open(image_path).convert("RGB")
        # 生成嵌入
        img_emb = IMAGE_MODEL.encode(
            image, 
            convert_to_tensor=False,
            normalize_embeddings=False  # 手动归一化，避免模型内置归一化冲突
        )
        normalized_emb = normalize_embedding(img_emb)
        return normalized_emb
    except Exception as e:
        # 打印详细错误
        print(f"生成图像嵌入失败 {image_path}: {e}")
        import traceback
        traceback.print_exc()  # 打印完整栈信息
        return None

def get_file_hash(file_path, chunk_size=4096):
    """计算文件内容的MD5哈希值（用于精准判断重复）"""
    try:
        hash_obj = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"计算文件哈希失败 {file_path}: {e}")
        return None

def find_duplicate_pdfs(root_dir):
    """查找指定目录下的重复PDF文件"""
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"目录不存在: {root_dir}")
        return {}
    
    # 步骤1：按文件大小粗筛
    size_groups = {}
    for pdf_file in root_path.rglob("*.pdf"):
        if pdf_file.is_file():
            file_size = pdf_file.stat().st_size
            if file_size not in size_groups:
                size_groups[file_size] = []
            size_groups[file_size].append(str(pdf_file))
    
    # 步骤2：对大小相同的文件计算哈希
    duplicate_groups = {}
    for size, files in size_groups.items():
        if len(files) < 2:
            continue
        
        hash_groups = {}
        for file_path in files:
            file_hash = get_file_hash(file_path)
            if not file_hash:
                continue
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append(file_path)
        
        # 收集重复组
        for file_hash, dup_files in hash_groups.items():
            if len(dup_files) >= 2:
                duplicate_groups[file_hash] = dup_files
    
    return duplicate_groups

def remove_duplicate_pdfs(root_dir, force_delete=False):
    """删除指定目录下的重复PDF文件"""
    duplicate_groups = find_duplicate_pdfs(root_dir)
    if not duplicate_groups:
        print("未找到重复PDF文件")
        return {"kept": 0, "deleted": 0, "groups": 0}
    
    kept_count = 0
    deleted_count = 0
    groups_count = len(duplicate_groups)
    
    print(f"\n=== 找到 {groups_count} 组重复PDF文件 ===")
    
    for hash_val, dup_files in duplicate_groups.items():
        print(f"\n重复组 (哈希: {hash_val}):")
        for idx, file_path in enumerate(dup_files):
            print(f"  {idx+1}. {file_path}")
        
        # 选择保留的文件
        def get_file_priority(file_path):
            p = Path(file_path)
            create_time = p.stat().st_ctime
            depth = len(p.parts)
            has_suffix = 1 if "_1" in p.stem or "_2" in p.stem else 0
            return (create_time, depth, has_suffix)
        
        dup_files_sorted = sorted(dup_files, key=get_file_priority)
        keep_file = dup_files_sorted[0]
        delete_files = dup_files_sorted[1:]
        
        print(f"\n保留文件: {keep_file}")
        kept_count += 1
        
        # 处理删除
        if delete_files:
            print(f"待删除文件 ({len(delete_files)} 个):")
            for del_file in delete_files:
                print(f"  - {del_file}")
                if force_delete:
                    try:
                        os.remove(del_file)
                        deleted_count += 1
                        print(f"已删除")
                    except Exception as e:
                        print(f"删除失败: {e}")
                else:
                    print(f"预览模式，未删除（添加 --force 强制删除）")
    
    print(f"\n=== 处理完成 ===")
    print(f"重复组数: {groups_count}")
    print(f"保留文件数: {kept_count}")
    print(f"删除文件数: {deleted_count}")
    if not force_delete:
        print(f"提示：当前为预览模式，未实际删除任何文件！")
    
    return {
        "kept": kept_count,
        "deleted": deleted_count,
        "groups": groups_count
    }



def classify_paper_multilabel(text, topics, threshold=None):
    """
    多标签分类：返回所有相似度≥阈值的主题
    :param text: 论文文本
    :param topics: 候选主题列表
    :param threshold: 分类阈值
    :return: 匹配的主题列表（按相似度降序）
    """
    if not text or len(topics) == 0:
        return []
    
    # 阈值优先级：传入值 > 全局配置
    cls_threshold = threshold if threshold is not None else CLASSIFICATION_THRESHOLD
    
    # 主题完整语义描述
    topic_descriptions = {
        "CV": "Computer Vision, image processing, visual recognition, CNN, ViT, image understanding",
        "NLP": "Natural Language Processing, text processing, LLMs, transformers, language understanding",
        "RL": "Reinforcement Learning, policy gradient, Q-learning, reward function, agent training",
        "VQA": "Visual Question Answering, multimodal reasoning, image question answering, MLLM in VQA, vision language interaction"
    }
    filtered_descriptions = {t: topic_descriptions.get(t, t) for t in topics}
    
    # 分段处理论文文本
    max_chunk_length = 2000
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    paper_embeddings = TEXT_MODEL.encode(text_chunks, convert_to_tensor=False)
    
    similarities = {}
    for topic, desc in filtered_descriptions.items():
        topic_embedding = TEXT_MODEL.encode(desc, convert_to_tensor=False)
        
        # 计算所有分段的平均相似度
        chunk_similarities = []
        for paper_emb in paper_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(paper_emb),
                torch.tensor(topic_embedding),
                dim=0
            ).item()
            chunk_similarities.append(similarity)
        
        avg_similarity = sum(chunk_similarities) / len(chunk_similarities)
        
        similarities[topic] = avg_similarity
    
    # 筛选符合阈值的主题
    matched_topics = [
        topic for topic, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        if sim >= cls_threshold
    ]
    
    # 兜底：无匹配时返回相似度最高的1个
    if not matched_topics:
        matched_topics = [max(similarities, key=similarities.get)]
    
    return matched_topics




def get_paper_file_list(topic_filter="", query_filter="", top_k=100):
    """
    获取符合条件的论文文件列表
    :param topic_filter: 主题筛选（如"VQA"）
    :param query_filter: 关键词语义筛选
    :param top_k: 返回最大数量
    :return: 去重后的文件路径列表
    """
    file_list = []
    
    # 1. 语义查询优先
    if query_filter:
        query_embedding = get_text_embedding(query_filter)
        if not query_embedding:
            return []
        
        results = PAPER_COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        for metadata in results['metadatas'][0]:
            if "copied_paths" in metadata:
                copied_paths_str = metadata["copied_paths"]
                if copied_paths_str:
                    copied_paths = copied_paths_str.split("||")
                    for cp in copied_paths:
                        file_path = cp.split(": ")[1] if ": " in cp else cp
                        file_list.append(file_path)
            else:
                file_list.append(metadata["path"])
    
    # 2. 主题筛选（无查询时）
    elif topic_filter:
        try:
            # 从数据库筛选
            all_papers = PAPER_COLLECTION.get(
                where={"topics": {"$contains": topic_filter}}
            )
            for metadata in all_papers["metadatas"]:
                if "copied_paths" in metadata:
                    copied_paths = metadata["copied_paths"].split("||")
                    for cp in copied_paths:
                        if topic_filter in cp:
                            file_path = cp.split(": ")[1] if ": " in cp else cp
                            file_list.append(file_path)
                else:
                    file_list.append(metadata["path"])
        except:
            # 降级：遍历主题目录
            topic_dir = PAPER_ROOT / topic_filter
            if topic_dir.exists():
                for pdf_file in topic_dir.glob("*.pdf"):
                    file_list.append(str(pdf_file))
    
    # 3. 无筛选：返回所有论文
    else:
        for topic_dir in PAPER_TOPICS.values():
            if topic_dir.exists():
                for pdf_file in topic_dir.glob("*.pdf"):
                    file_list.append(str(pdf_file))
    
    # 去重
    return list(set(file_list))


def get_clip_text_embedding(text):
    """基于CLIP模型生成文本向量"""
    if not text or len(text.strip()) == 0:
        return None
    try:
        text_emb = IMAGE_MODEL.encode(text.strip(), convert_to_tensor=False)
        return normalize_embedding(text_emb)  # 新增归一化
    except Exception as e:
        print(f"CLIP文本向量生成失败: {e}")
        return None
    

def normalize_embedding(embedding):
    """L2归一化向量（兼容多维数组，展平为一维）"""
    if embedding is None:
        return None
    
    # 将embedding转为numpy数组，展平为一维
    emb_array = np.array(embedding)
    if emb_array.ndim > 1:
        emb_array = emb_array.flatten()  # 多维→一维
    if len(emb_array) == 0:
        return None
    
    # L2归一化
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32)
    norm = torch.norm(emb_tensor, p=2, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)  # 防止除以0
    normalized_emb = (emb_tensor / norm).squeeze().tolist()
    
    return normalized_emb