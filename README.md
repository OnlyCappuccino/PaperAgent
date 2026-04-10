# Local Multi-Agent Research Assistant

一个适合边学边补功能的 **本地多智能体文献研究助理** 代码框架。

## 1. 项目目标

本项目用于搭建一个本地运行的文献问答系统，核心链路如下：

1. 批量读取 PDF / Markdown / TXT 文档
2. 对文档进行分块与向量化
3. 写入本地 ChromaDB
4. 基于 Query 检索相关片段
5. 由 Summarizer 生成结构化回答
6. 由 Critic 审核回答是否存在幻觉或证据不足
7. 若审核不通过，则根据 Critic 意见重写

## 2. 当前框架特点

- 已划分清晰模块，便于你逐步补功能
- 保留了大量 `TODO` 注释，适合学习式开发
- 默认采用 **OpenAI Compatible API**，兼容：
  - Ollama
  - vLLM
  - 本地 OpenAI 风格服务
- 默认采用：
  - PyMuPDF 处理 PDF
  - sentence-transformers 生成 embeddings
  - ChromaDB 做本地向量库
  - FastAPI 提供接口

## 3. 目录结构

```text
local_multi_agent_research_assistant/
├── app/
│   ├── agents/
│   ├── api/
│   ├── core/
│   ├── ingestion/
│   ├── llm/
│   ├── prompts/
│   ├── schemas/
│   ├── vectorstore/
│   └── workflow/
├── data/
│   ├── papers/
│   ├── chroma/
│   └── eval/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. 安装

建议 Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scriptsctivate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

核心配置说明：

- `LLM_BASE_URL`：本地模型 API 地址
- `LLM_MODEL_NAME`：本地模型名称
- `EMBEDDING_MODEL_NAME`：本地 embedding 模型名称
- `CHROMA_DIR`：ChromaDB 持久化目录
- `DOCS_DIR`：论文目录

## 6. 运行顺序

### 第一步：准备文档
把 PDF / md / txt 文件放进：

```text
data/papers/
```

### 第二步：建立索引

```bash
python -m scripts.build_index
```

### 第三步：命令行提问

```bash
python scripts/ask.py --query "这篇论文的核心方法是什么？"
```

### 第四步：启动 API

```bash
uvicorn app.api.server:app --reload
```

接口示例：

- `GET /health`
- `POST /index`
- `POST /ask`

## 7. 建议你优先补的功能

### 第一阶段
- 跑通本地 LLM API
- 能完成一次检索 + 总结
- 能返回引用片段

### 第二阶段
- 优化 chunk 策略
- 改进 prompt
- 增加 critic JSON 容错

### 第三阶段
- 增加评测脚本
- 增加 Web 前端
- 支持多轮会话记忆
- 支持论文元数据过滤

## 8. 推荐你怎么学习这个框架

建议按这个顺序阅读代码：

1. `app/config.py`
2. `app/llm/client.py`
3. `app/ingestion/loaders.py`
4. `app/vectorstore/chroma_store.py`
5. `app/agents/retriever.py`
6. `app/agents/summarizer.py`
7. `app/agents/critic.py`
8. `app/workflow/engine.py`
9. `app/api/server.py`

## 9. 你接下来最值得补的点

- 引文级别的证据绑定
- 更稳健的 critic 输出解析
- 多轮会话和历史状态管理
- 检索评测与回答评测
- 遥感领域专用 prompt / 术语模板
