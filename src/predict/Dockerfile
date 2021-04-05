FROM python:3.7-slim-buster

WORKDIR /app

RUN apt-get update && pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install torch==1.8.1
RUN pip install tensorflow==2.4.0
RUN pip install transformers==4.1.1

# Clear cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*1

COPY ./predict /app/predict
COPY ./tokcl/xmlcode.py /app/tokcl/xmlcode.py

ENTRYPOINT ["python", "-m", "predict.smtag"]


