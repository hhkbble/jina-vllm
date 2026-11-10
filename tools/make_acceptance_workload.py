"""Generate small semantic probes and the official capacity boundary workload."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        dict(
            id="short",
            query="How can I reduce inference latency?",
            documents=[
                "Use optimized GPU kernels, continuous batching and quantization.",
                "Garden flowers require sunlight and water.",
                "",
            ],
        ),
        dict(
            id="window",
            query="What keeps sliding and global attention distinct?",
            documents=[
                "sliding windows preserve local context. " * 600,
                "Global attention sees the entire causal history. " * 200,
            ],
        ),
        dict(
            id="count126",
            query="Which passage is relevant to GPU inference?",
            documents=[
                f"Passage {i}: "
                + (
                    "GPU inference optimization."
                    if i % 7 == 0
                    else "Some unrelated notes about gardening."
                )
                for i in range(126)
            ],
        ),
        dict(id="capacity", query=" a" * 20, documents=[" b" * 8192] * 16),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case) + "\n" for case in cases))


if __name__ == "__main__":
    main()
