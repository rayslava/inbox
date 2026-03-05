# inbox image build helpers
#
# Examples:
#   make images
#   make image-amd64
#   make push REGISTRY=ghcr.io/acme TAG=v1.2.3
#   make manifest REGISTRY=ghcr.io/acme TAG=v1.2.3

IMAGE ?= inbox
TAG ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo latest)
REGISTRY ?=
CARGO := $(or $(wildcard $(HOME)/.cargo/bin/cargo),$(shell command -v cargo))

ARCHES := amd64 arm64
TRIPLE_amd64 := x86_64-unknown-linux-musl
TRIPLE_arm64 := aarch64-unknown-linux-musl
PLATFORM_amd64 := linux/amd64
PLATFORM_arm64 := linux/arm64

IMG := $(if $(REGISTRY),$(REGISTRY)/$(IMAGE),$(IMAGE))

.PHONY: images push manifest clean

images: image-amd64 image-arm64

image-%:
	@triple="$(TRIPLE_$*)"; platform="$(PLATFORM_$*)"; \
	if [ -z "$$triple" ] || [ -z "$$platform" ]; then \
	  echo "ERROR: unsupported arch '$*' (supported: $(ARCHES))"; exit 1; \
	fi; \
	$(CARGO) zigbuild --release --target "$$triple" --bin inbox; \
	cd "target/$$triple/release" && \
	tar cf - inbox | docker import \
	  --platform "$$platform" \
	  --change 'EXPOSE 8080 9090' \
	  --change 'ENTRYPOINT ["/inbox"]' \
	  --change 'CMD ["--config", "/config/config.toml"]' \
	  - "$(IMG):$(TAG)-$*"; \
	echo "-> $(IMG):$(TAG)-$*"

push: images
	@test -n "$(REGISTRY)" || (echo "ERROR: REGISTRY is not set" && exit 1)
	@for arch in $(ARCHES); do docker push "$(IMG):$(TAG)-$$arch"; done

manifest: push
	docker buildx imagetools create --tag "$(IMG):$(TAG)" \
	  "$(IMG):$(TAG)-amd64" \
	  "$(IMG):$(TAG)-arm64"
	@if [ "$(TAG)" != "latest" ]; then \
	  docker buildx imagetools create --tag "$(IMG):latest" \
	    "$(IMG):$(TAG)-amd64" \
	    "$(IMG):$(TAG)-arm64"; \
	fi

clean:
	$(CARGO) clean

.DEFAULT_GOAL := images
