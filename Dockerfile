FROM python:3.10-slim
WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 5041
CMD ["uv", "run", "main.py"]