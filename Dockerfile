FROM tensorflow/tensorflow:latest-gpu
COPY . /east_tf2
WORKDIR /east_tf2
RUN apt-get update
RUN apt-get install -y python-dev
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt