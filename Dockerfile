FROM python:3.7-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
EXPOSE 88
ENV NOM Antonio_Bernardo
CMD ["python3", "app.py"]
