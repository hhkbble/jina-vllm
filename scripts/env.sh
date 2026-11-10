#!/usr/bin/env bash
# Source this file before setup or development commands.
export JINA_VLLM_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export JINA_VLLM_OUTPUT_DIR="$JINA_VLLM_PROJECT_ROOT/.local/output"
export TMPDIR="$JINA_VLLM_PROJECT_ROOT/.local/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export UV_CACHE_DIR="$JINA_VLLM_PROJECT_ROOT/.local/cache/uv"
export UV_PYTHON_INSTALL_DIR="$JINA_VLLM_PROJECT_ROOT/.local/python"
export TORCH_HOME="$JINA_VLLM_PROJECT_ROOT/.local/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$JINA_VLLM_PROJECT_ROOT/.local/cache/torchinductor"
export TRITON_CACHE_DIR="$JINA_VLLM_PROJECT_ROOT/.local/cache/triton"
export VLLM_CACHE_ROOT="$JINA_VLLM_PROJECT_ROOT/.local/cache/vllm"
export CUDA_CACHE_PATH="$JINA_VLLM_PROJECT_ROOT/.local/cache/cuda"

mkdir -p -- "$TMPDIR" "$JINA_VLLM_OUTPUT_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
    "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH"
