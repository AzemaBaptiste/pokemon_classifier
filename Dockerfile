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
COPY chicken.jpeg ./

RUN make install

# Cache resnet50 in the image
RUN bazema_pokemon --image_path chicken.jpeg

ENTRYPOINT ["bazema_pokemon"]