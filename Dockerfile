FROM python:3.8

WORKDIR /app

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bazema_pokemon/ ./bazema_pokemon
COPY Makefile ./
COPY setup.py ./
COPY README.md ./
COPY MANIFEST.in ./

RUN make install

ENTRYPOINT ["bazema_pokemon"]