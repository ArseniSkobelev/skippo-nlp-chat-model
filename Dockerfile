FROM python:3.8

WORKDIR /app

COPY . /app

ENV THRESHOLD=0.4

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["main.py"]