IMAGE_NAME = muniverse-test
TAG ?= scd
REGISTRY = docker.io/pranavm19

# Full image tag
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(TAG)

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--platform linux/amd64 \
		--cache-from=$(FULL_IMAGE) \
		-t $(FULL_IMAGE) .

push:
	docker push $(FULL_IMAGE)

pull-singularity:
	singularity pull docker://$(FULL_IMAGE)

run:
	docker run --gpus all -it $(FULL_IMAGE)

shell:
	docker run --gpus all -it $(FULL_IMAGE) /bin/bash
