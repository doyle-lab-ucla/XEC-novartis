REGISTRY ?= docker-repo.nibr.novartis.net/mcds/actlearn
TAG ?= latest

build:
	@echo Building docker image: $(REGISTRY):$(TAG)
	@docker build \
		-t $(REGISTRY):$(TAG) \
		-f Dockerfile .

debug:
	@docker run -it --rm \
		$(REGISTRY):$(TAG)
		
push:
	@echo Pushing docker image: $(REGISTRY):$(TAG)
	@docker push $(REGISTRY):$(TAG)