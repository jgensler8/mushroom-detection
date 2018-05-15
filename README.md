# mushroom classifier

Based on [this github repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10).

## Running

### Build the Container

```
docker build -t mushroom-classifier .
```

### Running

```
docker run --rm -it -v $PWD:/mushrooms --workdir /mushrooms mushroom-classifier /tensorflow/models/research/xml_to_csv.py

docker run --rm -it -v $PWD:/mushrooms mushroom-classifier ./generate_tfrecord.py \
  --csv_input /mushrooms/images/train_labels.csv \
  --image_dir /mushrooms/images/train \
  --output_path /mushrooms/train.record
  
docker run --rm -it -v $PWD:/mushrooms --workdir /tensorflow/models/research/object_detection mushroom-classifier ./train.py \
  --logtostderr \
  --train_dir=/mushrooms/images \
  --pipeline_config_path=/mushrooms/mushroomnet.config

docker run --rm -it -v $PWD:/mushrooms --workdir /tensorflow/models/research/object_detection mushroom-classifier ./export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path=/mushrooms/mushroomnet.config \
  --trained_checkpoint_prefix /mushrooms/images/model.ckpt-200 \
  --output_directory /mushrooms/inference_graph

docker run --rm -it -v $PWD:/mushrooms --workdir /tensorflow/models/research/object_detection mushroom-classifier ./Object_detection_image.py
```
