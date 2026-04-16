# Local Multi-Agent Research Assistant

这个 README 主要是给你“忘了命令时直接复制”的。

## 快速启动（最常用）

```powershell
# 1) 进入项目根目录
cd C:\Users\Cappuccino\Desktop\实习项目\local_multi_agent_research_assistant

# 2) 激活虚拟环境（如果还没创建，先看下方“首次安装”）
.venv\Scripts\Activate.ps1

# 3) 构建索引（推荐模块化运行）
python -m scripts.build_index

# 4) 命令行提问
python scripts\ask.py --query "这篇论文的核心方法是什么？"

# 5) 启动 API
uvicorn app.api.server:app --reload --host 127.0.0.1 --port 8000
```

---

## 首次安装

```powershell
cd C:\Users\Cappuccino\Desktop\实习项目\local_multi_agent_research_assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 环境变量（.env）

项目读取的是 `app/config.py` 里的配置项。建议在项目根目录放一个 `.env` 文件（你已经有 `.env` 就不用新建）。

最小必填建议：

```env
llm_base_url=http://127.0.0.1:11434/v1
llm_api_key=EMPTY
llm_model_name=qwen2.5:7b-instruct

embedding_base_url=http://127.0.0.1:11434/v1
embedding_api_key=EMPTY
embedding_model_name=BAAI/bge-m3

reranker_model=BAAI/bge-reranker-base

chroma_collection=research_chunks
chroma_dir=./data/chroma
docs_dir=./data/papers

top_k=5
chunk_size=800
chunk_overlap=120
max_rewrite_rounds=2
request_timeout=120
```

## 文档放置位置

把你的 `pdf/md/txt` 放到：

```text
data/papers
```

## 索引命令（你最容易忘的）

```powershell
# 普通构建
python -m scripts.build_index

# 指定 collection
python -m scripts.build_index --collection my_collection

# 仅重建当前 collection
python -m scripts.build_index --collection my_collection --rebuild

# 清理其他 collection（谨慎）
python -m scripts.build_index --collection my_collection --clear
```

## 问答命令

```powershell
# CLI 问答
python scripts\ask.py --query "请总结这篇论文的贡献点"
```

## API 启动与调用

启动：

```powershell
uvicorn app.api.server:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/health"
```

提问：

```powershell
$body = @{ query = "这篇论文的创新点是什么？" } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/ask" -ContentType "application/json" -Body $body
```

重建索引（API）：

```powershell
$body = @{ collection = "research_chunks"; clear = $false; rebuild = $false } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/index" -ContentType "application/json" -Body $body
```

## 常见报错速查

### 1) `ModuleNotFoundError: No module named 'app'`

原因：不在项目根目录运行，或不是模块化执行。  
解决：

```powershell
cd C:\Users\Cappuccino\Desktop\实习项目\local_multi_agent_research_assistant
python -m scripts.build_index
```

### 2) `Repo id must use alphanumeric ... D:\...\model.gguf`

原因：把本地文件路径当成 HuggingFace `repo_id` 传入。  
解决：如果你走 OpenAI-compatible embedding 服务，只需要配 `embedding_base_url + embedding_model_name`，不要传磁盘路径给 HuggingFace 加载逻辑。

### 3) `Error 500: input (...) is too large to process`

原因：单次 embedding 输入太大或服务批大小限制。  
解决：减小 `chunk_size`、降低一次请求文本量，或调整你本地 embedding 服务的 batch 配置。

## 开发常用命令

```powershell
# 运行测试
pytest -q

# 语法检查（关键文件）
python -m py_compile app\api\server.py app\agents\retriever.py app\workflow\engine.py
```

## 项目结构（核心）

```text
app/
  agents/        # Retriever / Summarizer / Critic
  api/           # FastAPI 入口
  core/          # 日志、引用处理
  ingestion/     # 加载与切块
  llm/           # OpenAI-compatible 客户端
  schemas/       # Pydantic 数据结构
  vectorstore/   # Chroma / Embedding / BM25 / Reranker
  workflow/      # 主流程编排
scripts/
  build_index.py
  ask.py
tests/
```
