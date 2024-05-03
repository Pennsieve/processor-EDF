FROM python:3.10.13

WORKDIR /processor

RUN apt clean && apt-get update

COPY processor/requirements.txt /processor/requirements.txt

RUN pip install -r /processor/requirements.txt

COPY processor/ /processor

CMD ["python3.10", "/processor/main.py" ]
