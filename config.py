import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import warnings


# 忽略无关警告
warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_MODEL_CACHE = {}
def get_model(model_name, device):
    """单例模式加载模型，避免重复初始化"""
    key = f"{model_name}_{device}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]



# 模型初始化）
TEXT_MODEL = get_model('all-MiniLM-L6-v2', DEVICE)
CLIP_MODEL = get_model('clip-ViT-B-32', DEVICE)

TEXT_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
IMAGE_MODEL = SentenceTransformer('clip-ViT-B-32', device=DEVICE)

# 路径配置
CUSTOM_ROOT = Path("D:/ai-agent")  # 自定义存储根路径
CLASSIFICATION_THRESHOLD = 0.4     # 多分类阈值

# Chroma数据库初始化
CHROMA_CLIENT = chromadb.PersistentClient(
    path=str(CUSTOM_ROOT / "chroma_db"),
    settings=Settings(anonymized_telemetry=False)
)

# 数据库集合
PAPER_COLLECTION = CHROMA_CLIENT.get_or_create_collection("papers")
IMAGE_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="images",
    metadata={"hnsw:space": "cosine"}  # 指定余弦相似度作为距离函数
)

# 论文分类目录
PAPER_ROOT = CUSTOM_ROOT / "papers"
PAPER_TOPICS = {
    "CV": PAPER_ROOT / "CV",
    "NLP": PAPER_ROOT / "NLP",
    "RL": PAPER_ROOT / "RL",
    "VQA": PAPER_ROOT / "VQA",
}

# 图片存储路径
IMAGE_ROOT = CUSTOM_ROOT / "images"

# 初始化目录
for path in PAPER_TOPICS.values():
    path.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT.mkdir(exist_ok=True)