FROM python:3.9-slim

COPY chat_app /home/frontend/chat_app

COPY requirements.txt /home/requirements.txt 

RUN pip install --no-cache-dir -r /home/requirements.txt 

RUN rm /home/requirements.txt

ENV PYTHONPATH="/home/frontend/:${PYTHONPATH:-}"


EXPOSE 8520

WORKDIR /home/frontend

CMD ["streamlit", "run", "chat_app/app.py", "--server.port=8520", "--server.address=0.0.0.0"]