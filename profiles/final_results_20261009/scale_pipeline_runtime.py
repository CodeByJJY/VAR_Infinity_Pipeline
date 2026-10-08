import numpy as np
import torch
import torch.nn.functional as F

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import (
    sample_with_top_k_top_p_also_inplace_modifying_logits_,
)

from kv_cache_state import (
    kv_modules,
    load_cfg_kv_group_consume,
    split_cfg_kv_cache_release,
)

from pipeline_state import (
    PipelineSampleState,
    split_cfg_tensor,
    merge_cfg_tensor,
    split_ca_kv,
    merge_ca_kv,
    split_logical_tensor,
    merge_logical_tensor,
)


class ScalePipelineRuntime:
    def __init__(
        self,
        model,
        vae,
        scale_schedule,
        cfg_list,
        tau_list,
        top_k=900,
        top_p=0.97,
        cfg_insertion_layer=(0,),
    ):
        self.model = model
        self.vae = vae
        self.scale_schedule = list(scale_schedule)

        if not isinstance(cfg_list, (list, tuple)):
            cfg_list = [cfg_list] * len(self.scale_schedule)
        if not isinstance(tau_list, (list, tuple)):
            tau_list = [tau_list] * len(self.scale_schedule)

        self.cfg_list = list(cfg_list)
        self.tau_list = list(tau_list)
        self.top_k = top_k
        self.top_p = top_p

        assert len(self.cfg_list) >= len(self.scale_schedule)
        assert len(self.tau_list) >= len(self.scale_schedule)

        # Current experiments use CFG, therefore the physical batch is 2B:
        # [cond_0 ... cond_B-1, uncond_0 ... uncond_B-1]
        self.use_cfg = any(np.asarray(self.cfg_list) != 1)
        assert self.use_cfg, "prototype runtime currently expects CFG != 1"

        if model.apply_spatial_patchify:
            self.vae_scale_schedule = [
                (t, 2 * h, 2 * w)
                for t, h, w in self.scale_schedule
            ]
        else:
            self.vae_scale_schedule = self.scale_schedule

        self.abs_cfg_insertion_layers = []
        self.add_cfg_on_logits = False
        self.add_cfg_on_probs = False

        depth = len(model.unregistered_blocks)

        for item in cfg_insertion_layer:
            if item == 0:
                self.add_cfg_on_logits = True
            elif item == 1:
                self.add_cfg_on_probs = True
            elif item < 0:
                assert depth + item > 0
                self.abs_cfg_insertion_layers.append(depth + item)
            else:
                raise ValueError(
                    f"invalid cfg_insertion_layer={item}"
                )

        # We do not need the probability-CFG path for the current experiments.
        assert not self.add_cfg_on_probs

        self.num_scales = len(self.scale_schedule)

        assert model.use_bit_label
        assert hasattr(vae.quantizer, "lfq")

    @torch.no_grad()
    def init_states(
        self,
        label_B_or_BLT,
        B,
        g_seed=0,
        negative_label_B_or_BLT=None,
    ):
        model = self.model

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

        # Build unconditional / negative text half exactly as the original
        # autoregressive_infer_cfg().
        if negative_label_B_or_BLT is None:
            kv_compact_un = kv_compact.clone()

            total = 0
            for le in lens:
                kv_compact_un[total:total + le] = model.cfg_uncond[:le]
                total += le

            kv_compact = torch.cat(
                (kv_compact, kv_compact_un),
                dim=0,
            )

            cu_seqlens_k = torch.cat(
                (
                    cu_seqlens_k,
                    cu_seqlens_k[1:] + cu_seqlens_k[-1],
                ),
                dim=0,
            )

        else:
            (
                kv_compact_un,
                lens_un,
                cu_seqlens_k_un,
                max_seqlen_k_un,
            ) = negative_label_B_or_BLT

            kv_compact = torch.cat(
                (kv_compact, kv_compact_un),
                dim=0,
            )

            cu_seqlens_k = torch.cat(
                (
                    cu_seqlens_k,
                    cu_seqlens_k_un[1:] + cu_seqlens_k[-1],
                ),
                dim=0,
            )

            max_seqlen_k = max(
                max_seqlen_k,
                max_seqlen_k_un,
            )

        kv_compact = model.text_norm(kv_compact)

        sos = cond_BD = model.text_proj_for_sos(
            (
                kv_compact,
                cu_seqlens_k,
                max_seqlen_k,
            )
        )

        projected_kv = model.text_proj_for_ca(kv_compact)

        ca_kv = (
            projected_kv,
            cu_seqlens_k,
            max_seqlen_k,
        )

        bs = 2 * B

        last_stage = (
            sos.unsqueeze(1).expand(bs, 1, -1)
            + model.pos_start.expand(bs, 1, -1)
        )

        with torch.amp.autocast("cuda", enabled=False):
            cond_BD_or_gss = (
                model.shared_ada_lin(cond_BD.float())
                .float()
                .contiguous()
            )

        last_parts = split_cfg_tensor(last_stage, B)
        cond_parts = split_cfg_tensor(cond_BD, B)
        gss_parts = split_cfg_tensor(cond_BD_or_gss, B)
        ca_parts = split_ca_kv(ca_kv, B)

        empty_cache = [
            (None, None)
            for _ in kv_modules(model)
        ]

        device = next(model.parameters()).device

        states = []

        for i in range(B):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(g_seed) + i)

            states.append(
                PipelineSampleState(
                    sample_id=i,
                    scale_idx=0,
                    last_stage=last_parts[i],
                    summed_codes=None,
                    cond_BD=cond_parts[i],
                    cond_BD_or_gss=gss_parts[i],
                    ca_kv=ca_parts[i],
                    kv_cache=list(empty_cache),
                    rng_state=generator.get_state().clone(),
                )
            )

        return states

    def _clear_model_cache(self):
        for m in kv_modules(self.model):
            m.cached_k = None
            m.cached_v = None
            m.caching = False

    def _sample_per_request(self, logits_BlV, states):
        B = len(states)

        if self.model.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]

            logits_bits = logits_BlV.reshape(
                tmp_bs,
                -1,
                2,
            )

            outputs = []

            for i, state in enumerate(states):
                generator = torch.Generator(
                    device=logits_bits.device
                )
                generator.set_state(state.rng_state)

                out = (
                    sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        logits_bits[i:i+1].clone(),
                        rng=generator,
                        top_k=self.top_k or self.model.top_k,
                        top_p=self.top_p or self.model.top_p,
                        num_samples=1,
                    )[:, :, 0]
                )

                state.rng_state = generator.get_state().clone()
                outputs.append(out)

            idx_Bld = torch.cat(outputs, dim=0)

            idx_Bld = idx_Bld.reshape(
                B,
                tmp_seq_len,
                -1,
            )

            return idx_Bld

        raise NotImplementedError(
            "current pipeline prototype targets use_bit_label models"
        )

    @torch.no_grad()
    def run_scale_batch(self, states):
        assert len(states) > 0

        model = self.model
        vae = self.vae

        scale_indices = {s.scale_idx for s in states}
        assert len(scale_indices) == 1

        si = next(iter(scale_indices))
        assert 0 <= si < self.num_scales

        B = len(states)
        bs = 2 * B

        pn = self.scale_schedule[si]
        cfg = self.cfg_list[si]

        # Restore the selected requests into one physical microbatch.
        last_stage = merge_cfg_tensor(
            [s.last_stage for s in states]
        )

        cond_BD = merge_cfg_tensor(
            [s.cond_BD for s in states]
        )

        cond_BD_or_gss = merge_cfg_tensor(
            [s.cond_BD_or_gss for s in states]
        )

        ca_kv = merge_ca_kv(
            [s.ca_kv for s in states]
        )

        load_cfg_kv_group_consume(
            model,
            [s.kv_cache for s in states],
        )

        attn_fn = None

        if model.use_flex_attn:
            attn_fn = model.attn_fn_compile_dict.get(
                tuple(self.scale_schedule[:si + 1]),
                None,
            )

        layer_idx = 0

        # Transformer blocks.
        for block_idx, block_chunk in enumerate(model.block_chunks):
            if (
                model.add_lvl_embeding_only_first_block
                and block_idx == 0
            ):
                last_stage = model.add_lvl_embeding(
                    last_stage,
                    si,
                    self.scale_schedule,
                    need_to_pad=0,
                )

            if not model.add_lvl_embeding_only_first_block:
                last_stage = model.add_lvl_embeding(
                    last_stage,
                    si,
                    self.scale_schedule,
                    need_to_pad=0,
                )

            for module in block_chunk.module:
                last_stage = module(
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=attn_fn,
                    scale_schedule=self.scale_schedule,
                    rope2d_freqs_grid=model.rope2d_freqs_grid,
                    scale_ind=si,
                )

                if (
                    cfg != 1
                    and layer_idx in self.abs_cfg_insertion_layers
                ):
                    mixed = (
                        cfg * last_stage[:B]
                        + (1 - cfg) * last_stage[B:]
                    )

                    last_stage = torch.cat(
                        (mixed, mixed),
                        dim=0,
                    )

                layer_idx += 1

        # The self-attention cache now contains all tokens through this scale.
        new_cache_parts = split_cfg_kv_cache_release(
            model,
            B,
        )

        # Logits / CFG.
        if cfg != 1 and self.add_cfg_on_logits:
            logits_BlV = (
                model.get_logits(last_stage, cond_BD)
                .mul(1 / self.tau_list[si])
            )

            logits_BlV = (
                cfg * logits_BlV[:B]
                + (1 - cfg) * logits_BlV[B:]
            )
        else:
            logits_BlV = (
                model.get_logits(
                    last_stage[:B],
                    cond_BD[:B],
                )
                .mul(1 / self.tau_list[si])
            )

        # Per-request RNG makes regrouping order independent.
        idx_Bld = self._sample_per_request(
            logits_BlV,
            states,
        )

        assert pn[0] == 1

        idx_Bld = idx_Bld.reshape(
            B,
            pn[1],
            pn[2],
            -1,
        )

        if model.apply_spatial_patchify:
            idx_Bld = idx_Bld.permute(
                0, 3, 1, 2
            )

            idx_Bld = F.pixel_shuffle(
                idx_Bld,
                2,
            )

            idx_Bld = idx_Bld.permute(
                0, 2, 3, 1
            )

        idx_Bld = idx_Bld.unsqueeze(1)

        codes = vae.quantizer.lfq.indices_to_codes(
            idx_Bld,
            label_type="bit_label",
        )

        if all(s.summed_codes is None for s in states):
            summed_codes = None
        else:
            assert all(
                s.summed_codes is not None
                for s in states
            )

            summed_codes = merge_logical_tensor(
                [s.summed_codes for s in states]
            )

        if si != self.num_scales - 1:
            contribution = F.interpolate(
                codes,
                size=self.vae_scale_schedule[-1],
                mode=vae.quantizer.z_interplote_up,
            )

            if summed_codes is None:
                summed_codes = contribution
            else:
                summed_codes = summed_codes + contribution

            next_stage = F.interpolate(
                summed_codes,
                size=self.vae_scale_schedule[si + 1],
                mode=vae.quantizer.z_interplote_up,
            )

            next_stage = next_stage.squeeze(-3)

            if model.apply_spatial_patchify:
                next_stage = F.pixel_unshuffle(
                    next_stage,
                    2,
                )

            next_stage = next_stage.reshape(
                *next_stage.shape[:2],
                -1,
            )

            next_stage = torch.permute(
                next_stage,
                [0, 2, 1],
            )

            next_stage = model.word_embed(
                model.norm0_ve(next_stage)
            )

            next_stage = next_stage.repeat(
                2,
                1,
                1,
            )

            next_parts = split_cfg_tensor(
                next_stage,
                B,
            )

        else:
            if summed_codes is None:
                summed_codes = codes
            else:
                summed_codes = summed_codes + codes

            next_parts = [None] * B

        summed_parts = split_logical_tensor(
            summed_codes,
            B,
        )

        for i, state in enumerate(states):
            state.kv_cache = new_cache_parts[i]
            state.summed_codes = summed_parts[i]
            state.last_stage = next_parts[i]
            state.scale_idx = si + 1

        # Avoid keeping a second copy of the same group inside the model.
        self._clear_model_cache()

        return idx_Bld

    @torch.no_grad()
    def decode_states(self, states):
        assert len(states) > 0
        assert all(
            s.scale_idx == self.num_scales
            for s in states
        )
        assert all(
            s.summed_codes is not None
            for s in states
        )

        summed_codes = merge_logical_tensor(
            [s.summed_codes for s in states]
        )

        return self.vae.decode(
            summed_codes.squeeze(-3)
        )
