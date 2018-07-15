FROM tensorflow/tensorflow:1.8.0
RUN apt-get update && apt-get install -y git wget unzip libsm6 libxext6 libfontconfig1 libxrender1
WORKDIR /tensorflow
RUN git clone https://github.com/tensorflow/models.git
COPY util/requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /tensorflow/models/research
RUN wget -q https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
RUN unzip protoc-3.5.1-linux-x86_64.zip
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH=$PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim
# # should take ~15 minutes
# RUN wget -q http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
# RUN tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
# RUN chown -R root:root faster_rcnn_resnet101_coco_11_06_2017
COPY util/exporter.py /tensorflow/models/research/object_detection/exporter.py
ENTRYPOINT ["python"]
