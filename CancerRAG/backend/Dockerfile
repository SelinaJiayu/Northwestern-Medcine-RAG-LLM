FROM python:3.9-slim

COPY requirements.txt /home/requirements.txt 

RUN pip install --no-cache-dir -r /home/requirements.txt 

RUN rm /home/requirements.txt

COPY cancer_rag /home/backend/cancer_rag

ENV PYTHONPATH="/home/backend/:${PYTHONPATH:-}"

EXPOSE 8000

WORKDIR /home/backend

CMD ["uvicorn", "cancer_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
