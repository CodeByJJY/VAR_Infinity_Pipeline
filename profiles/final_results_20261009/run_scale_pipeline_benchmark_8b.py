import argparse
import time
import torch

from run_infinity_batch_repeat import (
    add_common_arguments,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    encode_prompt,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from scale_pipeline_runtime import ScalePipelineRuntime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--prompt", type=str, default="a photo of a red panda sitting on a wooden chair")
    parser.add_argument("--save_file", type=str, default="./tmp.jpg")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    tokenizer, text_encoder = load_tokenizer(
        t5_path=args.text_encoder_ckpt
    )
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    scale_schedule = dynamic_resolution_h_w[
        args.h_div_w_template
    ][args.pn]["scales"]
    scale_schedule = [
        (1, h, w)
        for (_, h, w) in scale_schedule
    ]

    B = args.batch_size

    text_cond = encode_prompt(
        tokenizer,
        text_encoder,
        args.prompt,
        batch_size=B,
    )

    runtime = ScalePipelineRuntime(
        infinity,
        vae,
        scale_schedule,
        cfg_list=args.cfg,
        tau_list=args.tau,
        top_k=900,
        top_p=0.97,
        cfg_insertion_layer=[args.cfg_insertion_layer],
    )

    print("PIPELINE_BENCHMARK_BEGIN")
    N = B
    assert N == 2, "benchmark expects --batch_size 2"
    policy = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
    text_cond_1 = encode_prompt(
        tokenizer, text_encoder, args.prompt, batch_size=1
    )

    def original_once(batch, cond, seed):
        infinity.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=cond,
            g_seed=seed,
            B=batch,
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            cfg_sc=3,
            cfg_list=runtime.cfg_list,
            tau_list=runtime.tau_list,
            top_k=900,
            top_p=0.97,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k=0.0,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            softmax_merge_topk=-1,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=0,
            gt_ls_Bl=None,
            inference_mode=True,
            sampling_per_bits=1,
        )

    def run_sequential(seed):
        for i in range(N):
            original_once(1, text_cond_1, seed + i)

    def run_fixed(seed):
        original_once(N, text_cond, seed)

    def run_scaleaware(seed):
        states = runtime.init_states(text_cond, B=N, g_seed=seed)
        for si in range(len(scale_schedule)):
            preferred = min(policy[si], N)
            for start in range(0, N, preferred):
                runtime.run_scale_batch(
                    states[start:min(start + preferred, N)]
                )
        imgs = runtime.decode_states(states)
        assert imgs.shape[0] == N

    def measure(fn, seed):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(seed)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return sec, N / sec, peak

    tests = [
        ("Sequential-B1", run_sequential),
        ("Fixed-B2", run_fixed),
        ("ScaleAware", run_scaleaware),
    ]

    results = {}
    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16, cache_enabled=True
        ):
            for name, fn in tests:
                print("BENCH_WARMUP_BEGIN", name)
                wsec, wips, wmem = measure(fn, args.seed + 10000)
                print(
                    "BENCH_WARMUP", name,
                    f"seconds={wsec:.6f}",
                    f"img_s={wips:.6f}",
                    f"peak_gb={wmem:.3f}",
                )
                rows = []
                for r in range(3):
                    sec, ips, mem = measure(fn, args.seed + 100 * r)
                    rows.append((sec, ips, mem))
                    print(
                        "BENCH_MEASURE", name, r + 1,
                        f"seconds={sec:.6f}",
                        f"img_s={ips:.6f}",
                        f"peak_gb={mem:.3f}",
                    )
                mean_sec = sum(x[0] for x in rows) / len(rows)
                mean_ips = sum(x[1] for x in rows) / len(rows)
                peak_gb = max(x[2] for x in rows)
                results[name] = mean_ips
                print(
                    "RESULT", name,
                    f"mean_s={mean_sec:.6f}",
                    f"mean_img_s={mean_ips:.6f}",
                    f"peak_gb={peak_gb:.3f}",
                )

    gain = (
        results["ScaleAware"] / results["Sequential-B1"] - 1.0
    ) * 100.0
    fixed_pct = (
        results["ScaleAware"] / results["Fixed-B2"]
    ) * 100.0
    print("SCALEAWARE_VS_SEQUENTIAL_GAIN_PCT", f"{gain:.2f}")
    print("SCALEAWARE_FIXED_B2_THROUGHPUT_PCT", f"{fixed_pct:.2f}")
    print("PIPELINE_BENCHMARK_OK")
