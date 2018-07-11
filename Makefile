
CONTAINER_NAME=mushroom-classifier
DOCKER_WORKDIR=/workdir
IMAGE_SRC_BUCKET_NAME=mushroom-images
BUCKET_PREFIX=https://storage.googleapis.com/
DATATURKS_UPLOAD_FILE=dataturks/dataturks_upload.txt
DATATURKS_CONVERSION_FILE=/dataturks_to_PascalVOC.py
DATATURKS_JSON_FILE=dataturks/mushroombot.json
DATATURKS_OUTPUT_DIRECTORY=data
CLOUD_ML_DATA_BUCKET=mushroom-images-cloud-ml

PIPELINE_CONFIG=util/mushroomnet.config
INFERENCE_GRAPH_OUTPUT_DIR=inference_graph

STEPS=1000
LABEL_MAP_FILE=util/mushroom_label_map.pbtxt
DETECT_IMAGE_INPUT=data/1527940868.jpeg
DETECT_IMAGE_OUTPUT=data/detected.jpeg
DETECT_SCORE_THRESHOLD=0.35

MODEL_DIR=.
TRAIN_DATA=${DATATURKS_OUTPUT_DIRECTORY}
EVAL_DATA=${DATATURKS_OUTPUT_DIRECTORY}

build_container:
	docker build -t ${CONTAINER_NAME} .

generate_dataturks_upload:
	echo "Making all objects in bucket public"
	gsutil iam ch allUsers:objectViewer gs://${IMAGE_SRC_BUCKET_NAME}
	echo "Generating list of uploads to ${DATATURKS_UPLOAD_FILE}"
	gsutil ls gs://${IMAGE_SRC_BUCKET_NAME} | sed 's!gs://!${BUCKET_PREFIX}!' > ${DATATURKS_UPLOAD_FILE}

download_dataturks_labels:
	echo "no api key yet, you'll have to download from the UI"
	echo "you labeled data should be downloaded as a JSON file"

generate_voc_from_dataturks: build_container
	mkdir -p ${DATATURKS_OUTPUT_DIRECTORY}
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		util/dataturks_to_PascalVOC.py \
		${DATATURKS_JSON_FILE} \
		${DOCKER_WORKDIR}/${DATATURKS_OUTPUT_DIRECTORY} \
		${DOCKER_WORKDIR}/${DATATURKS_OUTPUT_DIRECTORY}
	echo "~~~~~~~~~~~~~~ WARNING ~~~~~~~~~~~~~~"
	echo "If any images show up with errors, you'll want to remove them from the dataset at this time"
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

generate_tfrecord_from_voc: build_container
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		util/generate_tfrecord.py \
		--image_dir ${DATATURKS_OUTPUT_DIRECTORY} \
		--output_path ${DATATURKS_OUTPUT_DIRECTORY}/train.tfrecord
	
clean_object_detection:
	rm -rf ${DATATURKS_OUTPUT_DIRECTORY}/model.ckpt-*
	rm -rf ${DATATURKS_OUTPUT_DIRECTORY}/events.*
	rm -rf ${DATATURKS_OUTPUT_DIRECTORY}/checkpoint

object_detection:
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		/tensorflow/models/research/object_detection/train.py \
			--logtostderr \
			--train_dir=${DATATURKS_OUTPUT_DIRECTORY} \
			--pipeline_config_path=${PIPELINE_CONFIG}

clean_inference_graph:
	rm -rf ${INFERENCE_GRAPH_OUTPUT_DIR}

export_inference_graph:
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		/tensorflow/models/research/object_detection/export_inference_graph.py \
			--input_type image_tensor \
			--pipeline_config_path=${PIPELINE_CONFIG} \
			--trained_checkpoint_prefix ${DATATURKS_OUTPUT_DIRECTORY}/model.ckpt-${STEPS} \
			--output_directory ${INFERENCE_GRAPH_OUTPUT_DIR}

detect_image:
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		util/Object_detection_image.py \
			--label_map_file ${LABEL_MAP_FILE} \
			--input_image ${DETECT_IMAGE_INPUT} \
			--output_image ${DETECT_IMAGE_OUTPUT} \
			--score_threshold ${DETECT_SCORE_THRESHOLD}

upload_data_to_bucket:
	gsutil -m cp -r file://${DATATURKS_OUTPUT_DIRECTORY} gs://${CLOUD_ML_DATA_BUCKET}

train_local:
	gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir ${MODEL_DIR} \
    -- \
    --train-files ${TRAIN_DATA} \
    --eval-files ${EVAL_DATA} \
    --train-steps 1000 \
    --eval-steps 100

train_cloud:
	echo "gcloud blah"
