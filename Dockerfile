FROM python:3.9

RUN apt-get update -y && apt-get install -y libgl1
RUN apt-get update && apt install -y openssl

ADD . /app

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# COPY . .

CMD ["python", "app.py"]