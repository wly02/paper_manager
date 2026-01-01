import click
import os
import shutil
from chromadb.errors import NotFoundError
from pathlib import Path
from config import (
    CUSTOM_ROOT, CLASSIFICATION_THRESHOLD, PAPER_COLLECTION,
    IMAGE_COLLECTION, PAPER_ROOT, IMAGE_ROOT
)
from utils import (
    extract_pdf_text, get_text_embedding, get_image_embedding,
    remove_duplicate_pdfs,classify_paper_multilabel,get_paper_file_list,get_clip_text_embedding
)


@click.group()
def cli():
    """本地多模态AI文献与图像管理助手（支持多分类+文件索引+删除重复文档）"""
    pass

@cli.command(name="add_paper")
@click.argument('path', type=click.Path(exists=True))
@click.option('--topics', default="CV,NLP,RL,VQA", help='分类主题列表，用逗号分隔')
@click.option('--root', default=str(CUSTOM_ROOT), help='自定义存储根路径')
@click.option('--threshold', default=CLASSIFICATION_THRESHOLD, help='分类相似度阈值（0-1）')
def add_paper_command(path, topics, root, threshold):
    """添加并多分类论文文件"""
    file_path = Path(path)
    custom_root = Path(root)
    paper_root = custom_root / "papers"
    
    if file_path.suffix.lower() != '.pdf':
        print("仅支持PDF文件")
        return
    
    print(f"正在处理论文: {file_path.name}")
    text = extract_pdf_text(file_path)
    if not text:
        print("无法提取论文文本内容")
        return
    
    topic_list = [t.strip() for t in topics.split(',') if t.strip()]
    if not topic_list:
        print("主题列表不能为空")
        return
    
    # 多标签分类
    matched_topics = classify_paper_multilabel(text, topic_list, threshold)
    if not matched_topics:
        print("无法分类论文")
        return
    
    print(f"匹配到的主题: {', '.join(matched_topics)}")
    
    # 复制到所有匹配的主题目录
    copied_paths = []
    for topic in matched_topics:
        target_dir = paper_root / topic
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理重名
        target_path = target_dir / file_path.name
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
        
        shutil.copy2(file_path, target_path)
        copied_paths.append(f"{topic}: {target_path}")
        print(f"论文已复制到: {target_path}")
    
    # 添加到数据库
    embedding = get_text_embedding(text)
    if embedding:
        paper_id = f"paper_{file_path.stem}_{os.urandom(4).hex()}"
        copied_paths_str = "||".join(copied_paths)
        PAPER_COLLECTION.add(
            embeddings=[embedding],
            metadatas=[{
                "path": str(file_path),
                "copied_paths": copied_paths_str,
                "topics": ",".join(matched_topics),
                "name": file_path.name
            }],
            ids=[paper_id]
        )
        print("论文已添加到向量数据库")

@cli.command(name="search_paper")
@click.argument('query')
@click.option('--top-k', default=5, help='返回结果数量')
@click.option('--topic', default="", help='筛选指定主题的论文')
def search_paper_command(query, top_k, topic):
    """语义搜索论文（支持按主题筛选）"""
    query_embedding = get_text_embedding(query)
    if not query_embedding:
        print("生成查询向量失败")
        return
    
    results = PAPER_COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"\n=== 论文搜索结果 (Top {top_k}) ===")
    if not results['metadatas'][0]:
        print("未找到相关论文")
        return
    
    # 按主题筛选结果
    filtered_results = []
    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
        if topic and topic not in metadata['topics'].split(','):
            continue
        filtered_results.append((metadata, distance))
    
    if not filtered_results:
        print(f"未找到{topic}主题的相关论文")
        return
    
    # 显示结果
    for i, (metadata, distance) in enumerate(filtered_results):
        similarity = 1 - distance
        print(f"{i+1}. {metadata['name']}")
        print(f"原始路径: {metadata['path']}")
        print(f"分类主题: {metadata['topics']}")
        copied_paths = metadata.get("copied_paths", "").split("||")
        print(f"存储路径: {'; '.join(copied_paths)}")
        print(f"相似度: {similarity:.4f}\n")

@cli.command(name="list_files")
@click.option('--topic', default="", help='筛选指定主题的论文')
@click.option('--query', default="", help='按关键词语义筛选论文')
@click.option('--top-k', default=100, help='返回最大文件数量')
@click.option('--output', default="", help='将结果输出到文件')
def list_files_command(topic, query, top_k, output):
    """仅返回相关论文文件列表（轻量化索引）"""
    if not topic and not query:
        print("提示：可通过 --topic 或 --query 筛选文件，留空则返回所有论文")
    
    print("正在检索文件...")
    file_list = get_paper_file_list(topic_filter=topic, query_filter=query, top_k=top_k)
    
    if not file_list:
        print("未找到符合条件的论文文件")
        return
    
    print(f"\n=== 符合条件的文件列表（共{len(file_list)}个）===")
    for idx, file_path in enumerate(file_list, 1):
        print(f"{idx}. {file_path}")
    
    # 输出到文件
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
        print(f"\n文件列表已保存到: {output}")

@cli.command(name="remove_duplicates")
@click.option('--root', default=str(PAPER_ROOT), help='论文根目录')
@click.option('--force', is_flag=True, help='是否强制删除重复文件（默认仅预览）')
def remove_duplicates_command(root, force):
    """删除指定目录下的重复PDF文档"""
    print(f"开始扫描重复文件，目录: {root}")
    print(f"模式: {'强制删除' if force else '预览（仅显示不删除）'}")
    
    stats = remove_duplicate_pdfs(root, force_delete=force)
    
    print(f"\n=== 重复文件清理统计 ===")
    print(f"扫描到重复组数: {stats['groups']}")
    print(f"保留文件数: {stats['kept']}")
    print(f"删除文件数: {stats['deleted']}")

@cli.command(name="batch_organize")
@click.argument('folder_path', type=click.Path(exists=True, file_okay=False), default=str(PAPER_ROOT))
@click.option('--topics', default="CV,NLP,RL,VQA", help='分类主题列表，用逗号分隔')
@click.option('--root', default=str(CUSTOM_ROOT), help='自定义存储根路径')
@click.option('--threshold', default=CLASSIFICATION_THRESHOLD, help='分类相似度阈值')
def batch_organize_command(folder_path, topics, root, threshold):
    """
    批量多分类整理论文（修复文件不存在报错）
    """
    folder = Path(folder_path)
    custom_root = Path(root)
    paper_root = custom_root / "papers"
    
    pdf_files = list(folder.rglob("*.pdf"))
    if not pdf_files:
        print("未找到PDF文件，处理结束")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
    topic_list = [t.strip() for t in topics.split(',') if t.strip()]
    
    # 初始化统计
    stats = {
        "total_processed": 0,
        "duplicate_skipped": 0,
        "wrong_topic_deleted": 0,
        "matched_topic_kept": 0,
        "unknown": 0,
        "copy_failed": 0 
    }

    for pdf_file in pdf_files:
        stem_parts = pdf_file.stem.split('_')
        is_auto_suffix = False
        if len(stem_parts) >= 2 and stem_parts[-1].isdigit():
            try:
                suffix_num = int(stem_parts[-1])
                if 1 <= suffix_num <= 9:
                    is_auto_suffix = True
            except:
                pass
        if is_auto_suffix:
            stats["duplicate_skipped"] += 1
            continue

        if not pdf_file.exists():
            stats["unknown"] += 1
            continue

        text = extract_pdf_text(pdf_file)
        if not text:
            stats["unknown"] += 1
            continue
        
        matched_topics = classify_paper_multilabel(text, topic_list, threshold)
        if not matched_topics:
            stats["unknown"] += 1
            continue

        
        for topic in matched_topics:
            target_dir = paper_root / topic
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / pdf_file.name
            
            # 复制前再次检查源文件是否存在
            if pdf_file.exists() and not target_path.exists():
                try:
                    shutil.copy2(pdf_file, target_path)
                    stats["matched_topic_kept"] += 1
                except Exception:
                    stats["copy_failed"] += 1

        # 删除非匹配主题目录的文件
        for topic in topic_list:
            topic_dir = paper_root / topic
            if not topic_dir.exists():
                continue

            for old_file in topic_dir.glob(f"{pdf_file.stem}*.pdf"):
                # 非匹配主题 → 删除
                if topic not in matched_topics:
                    try:
                        if old_file.exists():
                            os.remove(old_file)
                            stats["wrong_topic_deleted"] += 1
                    except:
                        pass
                else:
                    if old_file != topic_dir / pdf_file.name and old_file.exists():
                        try:
                            os.remove(old_file)
                            stats["wrong_topic_deleted"] += 1
                        except:
                            pass

        # 删除非匹配主题的源文件
        original_topic = pdf_file.parent.name if pdf_file.parent.name in topic_list else None
        if original_topic and original_topic not in matched_topics:
            try:
                if pdf_file.exists():
                    os.remove(pdf_file)
                    stats["wrong_topic_deleted"] += 1
            except:
                pass

        stats["total_processed"] += 1


    print("\n处理完成，统计信息：")
    print(f"- 成功处理文件数：{stats['total_processed']}")
    print(f"- 删除非匹配主题/冗余文件数：{stats['wrong_topic_deleted']}")
    print(f"- 匹配主题保留文件数：{stats['matched_topic_kept']}")
    print(f"- 跳过冗余文件数：{stats['duplicate_skipped']}")
    print(f"- 复制失败文件数：{stats['copy_failed']}")
    print(f"- 未处理文件数：{stats['unknown']}")

@cli.command(name="search_image")
@click.argument('query')
@click.option('--top-k', default=5, help='返回结果数量')
def search_image_command(query, top_k):
    """语义搜索图片（最小改动修复相似度全0问题）"""
    query_embedding = get_clip_text_embedding(query)
    if not query_embedding:
        print("生成查询向量失败（CLIP模型）")
        return
    
    try:
        results = IMAGE_COLLECTION.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
    except Exception as e:
        print(f"查询失败：{e}")
        return
    
    print(f"\n=== 图片搜索结果 (Top {top_k}) ===")
    if not results['metadatas'][0]:
        print("未找到相关图片")
        return
    
    matched_results = []
    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):

        real_similarity = 1 - (distance / 2)
        real_similarity = min(1.0, real_similarity)
        matched_results.append((metadata, real_similarity))
    
    # 按相似度降序排序
    matched_results.sort(key=lambda x: x[1], reverse=True)
    
    # 显示结果（保留4位小数）
    for i, (metadata, similarity) in enumerate(matched_results):
        print(f"{i+1}. {metadata['name']}")
        print(f"   路径: {metadata['path']}")
        print(f"   相似度: {similarity:.4f}\n")

@cli.command(name="index_images")
@click.argument('image_folder', type=click.Path(exists=True, file_okay=False))
@click.option('--root', default=str(CUSTOM_ROOT), help='自定义存储根路径')
@click.option('--update', is_flag=True, help='已有图片时更新嵌入')
def index_images_command(image_folder, root, update):
    """索引图片文件夹中的所有图片（避免重复入库）"""
    folder = Path(image_folder)
    custom_root = Path(root)
    image_root = custom_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    
  
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f"*{ext}"))
    image_files = list(set(image_files))  # 去重
    
    if not image_files:
        print("未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始索引...")
    success_count = 0
    skip_count = 0  # 新增：跳过重复计数
    
    for image_file in image_files:
       
        try:
            # 查询数据库中路径匹配的记录
            existing_records = IMAGE_COLLECTION.get(where={"path": str(image_file)})
            if existing_records["ids"]:  # 已有该图片
                if update:
                  
                    IMAGE_COLLECTION.delete(ids=existing_records["ids"])
                    print(f"更新已有图片：{image_file.name}")
                else:
                    
                    skip_count += 1
                    continue
        except Exception as e:
         
            pass
        
    
        embedding = get_image_embedding(image_file)
        if embedding:
            image_id = f"image_{image_file.stem}_{os.urandom(4).hex()}"
            IMAGE_COLLECTION.add(
                embeddings=[embedding],
                metadatas=[{"path": str(image_file), "name": image_file.name}],
                ids=[image_id]
            )
            success_count += 1
    
    # 精简统计输出
    print("\n索引完成，统计信息：")
    print(f"- 新增/更新图片数：{success_count}")
    print(f"- 跳过重复图片数：{skip_count}")
    print(f"- 总计处理图片数：{success_count + skip_count}")


@cli.command(name="delete_all_images")
def delete_all_images_command():
    """删除数据库中所有图片的索引记录,仅删索引不删本地图片"""
    confirm = click.prompt("确定要删除数据库中所有图片索引吗？(输入 y 确认，其他取消)", type=str)
    if confirm.lower() not in ["y", "yes"]:
        print("取消删除操作")
        return
    
    try:
        # 获取所有图片索引的ID并删除
        all_records = IMAGE_COLLECTION.get()
        if all_records["ids"]:
            IMAGE_COLLECTION.delete(ids=all_records["ids"])
            print(f"成功删除 {len(all_records['ids'])} 个图片索引")
        else:
            print("数据库中无图片索引可删除")
    except NotFoundError:
        print("图片集合不存在，无需删除")
    except Exception as e:
        print(f"删除失败：{str(e)}")