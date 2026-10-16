# Multi-stage: build the React frontend, then ship it with the FastAPI backend
# so ONE container serves the UI (/) and the API (/api).
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx vite build

FROM python:3.12-slim
WORKDIR /srv/app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend /fe/dist ./static

ENV API_KEY=demo-key \
    LLM_PROVIDER=fake \
    DATABASE_URL=sqlite:///./data/app.db \
    DEMO_MODE=true \
    LOG_LEVEL=INFO \
    VITE_DEMO_MODE=true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
