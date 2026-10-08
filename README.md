# VAR Infinity Scale-Aware Multi-Sample Pipeline

Experimental **scale-aware multi-sample inference pipeline** for visual autoregressive generation, built on top of [FoundationVision/Infinity](https://github.com/FoundationVision/Infinity).

> **This repository is not the original Infinity project.**
> It is a research extension that adds scale-dependent batching, per-request execution state, per-sample KV-cache regrouping, and GPU benchmarking infrastructure to the upstream Infinity implementation.

## Overview

Infinity generates images over progressively larger visual scales. This project explores whether multiple independent generation requests can be dynamically grouped and regrouped across those scales to improve GPU utilization and inference throughput.

Implemented framework:
- Scale-aware multi-sample scheduling
- Dynamic regrouping between visual scales
- Per-request pipeline state
- Per-sample KV-cache split / regroup / restore
- Memory-optimized KV-cache consume / release
- Per-sample RNG-state preservation
- Scale-dependent batch-size policies
- Infinity-2B / Infinity-8B execution
- A100 / H100 benchmark infrastructure
- Real KV-cache probing and regroup validation

## Main Pipeline Components

```text
tools/
├── scale_pipeline_runtime.py
│   └── Core scale-aware multi-sample execution runtime
├── pipeline_state.py
│   └── Per-request execution and sampling state
├── kv_cache_state.py
│   └── KV-cache split, regroup, load, consume, and release
├── run_scale_pipeline_benchmark.py
│   └── Infinity-2B pipeline benchmark
├── run_scale_pipeline_benchmark_8b.py
│   └── Infinity-8B pipeline benchmark
├── run_scale_pipeline_smoke.py
│   └── End-to-end pipeline smoke test
├── run_scale_pipeline_regroup_smoke.py
│   └── Real KV-cache regroup validation
├── probe_real_kv.py
│   └── Runtime inspection of Infinity KV-cache structure
├── test_pipeline_state.py
│   └── Pipeline-state unit tests
└── test_sampling_rng_order.py
    └── Per-sample RNG ordering/equivalence tests
```

GPU execution scripts are under `jobs/`, profiling/benchmark outputs under `profiles/`, and generated experiment outputs under `outputs/`.

The upstream model code remains under `infinity/`. `infinity/models/infinity.py` also contains profiling/instrumentation changes required by this work.

## Pipeline Concept

Conventional sequential execution:

```text
Request 0 : S0 -> S1 -> S2 -> ... -> S12
Request 1 : S0 -> S1 -> S2 -> ... -> S12
Request 2 : S0 -> S1 -> S2 -> ... -> S12
Request 3 : S0 -> S1 -> S2 -> ... -> S12
```

Scale-aware execution:

```text
Scale 0   : [R0 R1 R2 R3]
Scale 1   : [R0 R1 R2 R3]
...
Scale k   : regroup / resize batch according to scale policy
...
Scale 12  : remaining active requests
```

Each request preserves its own sampling state, RNG state, KV cache, and generated history, allowing physical reordering/regrouping without losing per-request state.

## Final Benchmark Results

Post-memory-optimization measurements:

| GPU | Model | Sequential (img/s) | Fixed Batch (img/s) | Scale-Aware (img/s) | Gain vs Sequential | Scale-Aware / Fixed |
|---|---:|---:|---:|---:|---:|---:|
| A100 | Infinity-2B | 0.662895 | 0.892118 | 0.888419 | +34.02% | 99.59% |
| H100 | Infinity-2B | 0.959406 | 1.431940 | 1.385600 | +44.42% | 96.76% |
| A100 | Infinity-8B | 0.334422 | 0.370016 | 0.356381 | +6.57% | 96.32% |
| H100 | Infinity-8B | 0.518150 | 0.597204 | 0.572027 | +10.40% | 95.78% |

Detailed logs and summaries: `profiles/final_results_20261009/`

## Repository Structure

```text
VAR_Infinity_Pipeline/
├── infinity/       # Upstream Infinity implementation + required instrumentation
├── tools/          # Scale-aware pipeline runtime and experiments
├── jobs/           # A100/H100 SLURM benchmark scripts
├── profiles/       # Profiling logs and benchmark results
├── outputs/        # Generated experiment outputs
├── evaluation/     # Upstream Infinity evaluation code
├── scripts/        # Upstream training/evaluation scripts
└── data/           # Example/evaluation data
```

## Model Weights and Environment

Large binary artifacts are intentionally excluded from GitHub:

```text
weights/
.venv_inf/
```

The experiments use Infinity checkpoints together with the `flan-t5-xl` text encoder. See the upstream Infinity project for the original model release and checkpoint information.

## Upstream Project and Attribution

This work is built on top of **Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis**.

Original project: [FoundationVision/Infinity](https://github.com/FoundationVision/Infinity)

The base model architecture, model implementation, training/evaluation infrastructure, and upstream assets originate from the FoundationVision Infinity project. This repository focuses on the additional **scale-aware multi-sample inference pipeline, KV-cache state management, scheduling logic, and GPU benchmarking framework**.

If you use the Infinity model itself, please cite the original work:

```bibtex
@misc{Infinity,
    title={Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis},
    author={Jian Han and Jinlai Liu and Yi Jiang and Bin Yan and Yuqi Zhang and Zehuan Yuan and Bingyue Peng and Xiaobing Liu},
    year={2024},
    eprint={2412.04431},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

This repository retains the upstream `LICENSE` file. The original Infinity implementation is distributed under the MIT License. Please refer to `LICENSE` and the upstream project for the applicable terms and attribution.
