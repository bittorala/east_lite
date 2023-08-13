FROM tensorflow/tensorflow:2.7.0-gpu
COPY . /east_tf2

WORKDIR /east_tf2
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt

WORKDIR /east_tf2/spa
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt install -y nodejs
RUN npm install
RUN npm run build

EXPOSE 3000
EXPOSE 8000

ENV APP_RUNNING 1

WORKDIR /east_tf2
RUN chmod +x ./run_app.sh
CMD ./run_app.sh
