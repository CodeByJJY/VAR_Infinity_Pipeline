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

    print("PIPELINE_REGROUP_SMOKE_BEGIN")
    print("logical_batch =", B)
    print("num_scales =", len(scale_schedule))

    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=True,
            dtype=torch.bfloat16,
            cache_enabled=True,
        ):
            states = runtime.init_states(
                text_cond,
                B=B,
                g_seed=args.seed,
            )

            torch.cuda.synchronize()
            t0 = time.time()

            for si in range(len(scale_schedule)):
                s = time.time()

                group = states if si % 2 == 0 else list(reversed(states))
                print("PIPELINE_GROUP_ORDER", si, [x.sample_id for x in group])
                idx = runtime.run_scale_batch(group)

                torch.cuda.synchronize()

                print(
                    "PIPELINE_SCALE",
                    si,
                    "batch",
                    len(states),
                    "idx_shape",
                    tuple(idx.shape),
                    "next_scale_idx",
                    states[0].scale_idx,
                    "wall_ms",
                    f"{(time.time()-s)*1000:.3f}",
                )

            imgs = runtime.decode_states(states)
            torch.cuda.synchronize()

            elapsed = time.time() - t0

    print("PIPELINE_DECODE_SHAPE", tuple(imgs.shape))
    print("PIPELINE_TOTAL_S", f"{elapsed:.6f}")
    print("PIPELINE_FINAL_SCALE_IDX", states[0].scale_idx)

    assert states[0].scale_idx == len(scale_schedule)
    assert imgs.shape[0] == B

    print("PIPELINE_REGROUP_SMOKE_OK")
