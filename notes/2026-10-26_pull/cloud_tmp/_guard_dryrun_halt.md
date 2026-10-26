# VRAM guard halt — snapshot before anything was killed

- generated: **2026-10-25 14:42:33 北京** (2026-10-25T06:42:33Z)
- reason: free 7625 MiB < 99999 MiB for 2 consecutive reads at 2026-10-25 06:42:33 北京
- code: `f04df9ff697d0d4fb22d39a31f405a1def0789c1`  ·  disk free 38.7 GiB
- VRAM: free **7625** / used 40884 / total 49140 MiB, util 100%
- compute apps: `50313, 18678 MiB
50836, 22192 MiB`

## Processes at halt

- **run_9**: `50059       21:20 bash scripts/run_9.sh`
- **train**: `50313       21:16 /opt/conda/bin/python src/train.py --config configs/cloud48g_main.yaml --mu 1 --seed 0 --tag mu1_s0`
- **engine**: `50836       20:57 VLLM::EngineCore`
- **guard**: `49863       21:46 bash tools/vram_guard.sh
  55342       00:02 bash tools/vram_guard.sh`

## What is on disk

| run | summary | metrics rows | rollouts | state.pt | resp_len (last) | pass_rate | zero_var_frac | cost CNY |
|---|---|---|---|---|---|---|---|---|
| mu1_s0 | no | 1 | 2 | 798.4 MiB, 458s old | 292.925 | 0.218018 | 0.0 | 0.4494 |

## Current run `mu1_s0` — recent rollouts

| rollout_idx | resp_len_mean | pass_rate | zero_var_frac | cost CNY |
|---|---|---|---|---|
| 0 | 292.925 | 0.218018 | 0.0 | 0.4494 |

resp_len slope: **None** tokens/rollout (flat or falling)

```
(EngineCore pid=50836) INFO 10-25 06:21:54 [vllm.py:1929] Cudagraph is disabled under eager mode
(EngineCore pid=50836) INFO 10-25 06:21:54 [compilation.py:317] Enabled custom fusions: norm_quant, act_quant
(MainProcess pid=50313) INFO 10-25 06:21:54 [hf.py:648] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
[info] vocab=151669 params=1.790B trainable=69.7M
[info] est. logits memory per micro-batch ~0.14 GiB (logit_chunk=64)
[info] micro=4 group=8 -> 1024 forward chunks / 512 sigma groups per step
[info] arm mu=1 b=4096 Br=4096 micro=4 M=1024 rollouts=8 total_steps=8
(MainProcess pid=50313) WARNING 10-25 06:21:55 [input_processor.py:289] vLLM has deprecated support for supporting different tokenizers for different LoRAs. By default, vLLM uses base model's tokenizer. If you are using a LoRA with its own tokenizer, consider specifying `--tokenizer [lora_path]` to use the LoRA tokenizer.
[ids] r000.npz 1.1 MiB sha256=82bbd063c0dfcff6 (4096 resp, max_seq_len 626)
[r0 mu00] pass=0.218 r=+0.263 len=293 kl=+0.00e+00 clip=0.0000 gn=0.005 sig=0.06626 d=0.0723
[r0] done: rollout 285.5s oldlogp 124.3s grad 367.9s update 492.2s cum 4096 resp, 0.216 GPU-h, ¥0.45
[ckpt] runs/mu1_s0/state.pt @ rollout 0
```

## Decisions this snapshot was taken for (nothing has been run)

Resume always picks up from `state.pt`; the rollout in flight at halt is lost (state.pt is written every rollout, D-21).

```bash
# A. same config, same risk — resume the campaign where it stopped
setsid nohup bash scripts/run_9.sh >>/tmp/campaign_main.log 2>&1 &

# B. buy headroom with the reservation, ~+0.8 h over 9 runs
#    configs/cloud48g_util042.yaml: util 0.42 (frees ~1.4 GiB of the 21.3 GiB reservation)
CFG=configs/cloud48g_util042.yaml setsid nohup bash scripts/run_9.sh >>/tmp/campaign_util042.log 2>&1 &

# C. the old fallback: util 0.38 / max_num_seqs 192
CFG=configs/cloud48g_lowmem.yaml setsid nohup bash scripts/run_9.sh >>/tmp/campaign_lowmem.log 2>&1 &
```

Discount rate for sitting idle on the box: the instance bills whether or not it trains.
