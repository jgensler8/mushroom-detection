# This is so echo commands don't add extra newlines
SHELL=bash

CONTAINER_NAME=mushroom-classifier
DOCKER_WORKDIR=/workdir
IMAGE_SRC_BUCKET_NAME=mushroom-images
BUCKET_PREFIX=https://storage.googleapis.com/
DATATURKS_UPLOAD_FILE=dataturks/dataturks_upload.txt
DATATURKS_CONVERSION_FILE=/dataturks_to_PascalVOC.py
DATATURKS_JSON_FILE=dataturks/mushroombot.json
DATATURKS_OUTPUT_DIRECTORY=data
CLOUD_ML_DATA_BUCKET=mushroombot-ml

PIPELINE_CONFIG=util/mushroomnet.config
INFERENCE_GRAPH_OUTPUT_DIR=inference_graph

STEPS=1000
LABEL_MAP_FILE=util/mushroom_label_map.pbtxt
DETECT_IMAGE_INPUT=data/1527937205.jpeg
DETECT_IMAGE_OUTPUT=data/detected.jpeg
DETECT_SCORE_THRESHOLD=0.35

MODEL_DIR=.
TRAIN_DATA=${DATATURKS_OUTPUT_DIRECTORY}
EVAL_DATA=${DATATURKS_OUTPUT_DIRECTORY}
GCP_MODEL_NAME=mushroombot_detector
GCP_REGION=us-central1
GCP_MODEL_BINARIES=gs://${CLOUD_ML_DATA_BUCKET}/saved_model
GCP_MODEL_VERSION=v1
GCP_CLOUD_ML_JSON_INSTANCES=input.json
GCP_VERSION_CONFIG_FILE=util/model_version.yaml

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
			--input_type b64encoded_image_string_tensor \
			--pipeline_config_path=${PIPELINE_CONFIG} \
			--trained_checkpoint_prefix ${DATATURKS_OUTPUT_DIRECTORY}/model.ckpt-${STEPS} \
			--output_directory ${INFERENCE_GRAPH_OUTPUT_DIR}

debug_inference_graph:
	saved_model_cli show --dir ${INFERENCE_GRAPH_OUTPUT_DIR}/saved_model --all

detect_image:
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		util/Object_detection_image.py \
			--label_map_file ${LABEL_MAP_FILE} \
			--input_image ${DETECT_IMAGE_INPUT} \
			--output_image ${DETECT_IMAGE_OUTPUT} \
			--score_threshold ${DETECT_SCORE_THRESHOLD}

upload_data_to_bucket:
	gsutil -m cp -r file://${DATATURKS_OUTPUT_DIRECTORY} gs://${CLOUD_ML_DATA_BUCKET}

upload_model_to_bucket:
	gsutil -m cp -r file://${INFERENCE_GRAPH_OUTPUT_DIR}/saved_model gs://${CLOUD_ML_DATA_BUCKET}

create_model:
	gcloud ml-engine models create ${GCP_MODEL_NAME} --regions=${GCP_REGION}

create_model_version:
	gcloud ml-engine versions create ${GCP_MODEL_VERSION} \
	--model ${GCP_MODEL_NAME} \
	--config ${GCP_VERSION_CONFIG_FILE}

create_gcp_json_input:
	docker run --rm -it --workdir ${DOCKER_WORKDIR} -v $$PWD:${DOCKER_WORKDIR} ${CONTAINER_NAME} \
		util/GCP_detection_json.py \
		--input_image ${DETECT_IMAGE_INPUT} \
		--output_file ${GCP_CLOUD_ML_JSON_INSTANCES}

gcp_detect_image:
	gcloud ml-engine predict \
	  --model ${GCP_MODEL_NAME} \
	  --version ${GCP_MODEL_VERSION} \
	  --json-instances ${GCP_CLOUD_ML_JSON_INSTANCES} \
	  --verbosity debug

delete_model_version:
	gcloud ml-engine versions delete ${GCP_MODEL_VERSION} --model ${GCP_MODEL_NAME}
