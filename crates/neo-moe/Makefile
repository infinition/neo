# neo-moe Makefile
# =================
# Build targets for the MoE expert streaming engine.
#
# Windows: uses cargo directly (open neo-moe.sln in VS or cargo build)
# Linux:   full support including O_DIRECT + io_uring
#
# Usage:
#   make build      # cargo build --release (cdylib + lib)
#   make test       # cargo test
#   make bench      # cargo run --example bench -- --model /path/to/model.gguf
#   make python     # python qwen_neo_moe.py --model /path/to/model.gguf
#   make install    # copy the dll/so to Qwen_3.6_35b/

UNAME := $(shell uname -s)
MODEL ?= ../Qwen_3.6_35b/.models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

.PHONY: all build test bench python clean install

all: build

build:
	cargo build --release

test:
	cargo test --release

bench: build
	cargo run --release --example bench -- --model "$(MODEL)"

integration: build
	cargo run --release --example integration -- --model "$(MODEL)"

python: build
	python qwen_neo_moe.py --model "$(MODEL)"

demo: build
	python qwen_neo_moe.py --demo --model "$(MODEL)"

install: build
	@echo "==> Installing neo-moe shared library to Qwen_3.6_35b/"
	@mkdir -p ../Qwen_3.6_35b/neo-moe-bin/
ifeq ($(UNAME), Linux)
	cp target/release/libneo_moe.so ../Qwen_3.6_35b/neo-moe-bin/
	cp neo_moe_backend.h       ../Qwen_3.6_35b/neo-moe-bin/
	cp neo_moe_backend.c       ../Qwen_3.6_35b/neo-moe-bin/
else
	cp target/release/neo_moe.dll   ../Qwen_3.6_35b/neo-moe-bin/
	cp target/release/neo_moe.lib   ../Qwen_3.6_35b/neo-moe-bin/
	cp neo_moe_backend.h            ../Qwen_3.6_35b/neo-moe-bin/
	cp neo_moe_backend.c            ../Qwen_3.6_35b/neo-moe-bin/
endif
	@echo "==> Installed."

clean:
	cargo clean
