FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先安装依赖，后续仅代码变更可复用该层缓存
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 再拷贝业务代码，确保代码改动只影响后面层
COPY app /app/app
COPY scripts /app/scripts

EXPOSE 7090

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "7090"]
