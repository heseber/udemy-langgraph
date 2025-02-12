FROM langchain/langgraph-api:3.11



ADD . /deps/rag-langgraph-studio

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
ENV LANGSERVE_GRAPHS='{"rag": "/deps/rag-langgraph-studio/graph/graph_core.py:app"}'

WORKDIR /deps/rag-langgraph-studio