FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY ["pyproject.toml", "uv.lock" ,"./"]

RUN pip install .

COPY ["predict_test.py", "pipeline_v1.bin", "./"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict_test:app", "--host", "0.0.0.0", "--port", "8000"]