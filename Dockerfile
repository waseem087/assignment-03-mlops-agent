FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn==1.7.1 joblib dill pandas numpy
COPY api/ /app/api/
COPY models/approved/model.joblib /app/model/model.joblib
ENV APP_VERSION=v1.0.0
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
