#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$JINA_VLLM_PROJECT_ROOT"
mode="${1:-bf16}"
shift "$(( $# > 0 ? 1 : 0 ))"
if [[ "$mode" != bf16 && "$mode" != fp8 ]]; then
    echo "Usage: scripts/serve.sh [bf16|fp8] [vllm serve options...]" >&2
    exit 2
fi
export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}jina_reranker_v35"
if [[ -z "${JINA_MODEL_DIR:-}" ]]; then
    JINA_MODEL_DIR="$(uv run --locked python -c 'from huggingface_hub import snapshot_download; from jina_vllm.contract import MODEL_ID, REVISION; print(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))')"
fi
common_args=(
    "$JINA_MODEL_DIR"
    --host 127.0.0.1
    --served-model-name jina35
    --model-impl vllm
    --runner pooling
    --hf-overrides '{"architectures":["JinaRerankerV35"],"embedding_size":512}'
    --pooler-config '{"task":"token_embed"}'
    --dtype bfloat16
    --kv-cache-dtype bfloat16
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --max-model-len 131072
    --max-num-batched-tokens 8192
    --max-num-seqs 4
    --gpu-memory-utilization 0.75
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --disable-flashinfer-autotune
)
attention_args=(--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}')
for arg in "$@"; do
    case "$arg" in
        --attention-backend|--attention-backend=*|--attention-config|--attention-config=*|-ac|-ac.*|--attention-config.*)
            attention_args=()
            break
            ;;
    esac
done
common_args+=("${attention_args[@]}")
if [[ "$mode" == fp8 ]]; then
    common_args+=(
        --quantization fp8_per_channel
        --quantization-config '{"ignore":["model.embed_tokens","projector.*"]}'
        --linear-backend cutlass
    )
fi
exec uv run --locked vllm serve "${common_args[@]}" "$@"
