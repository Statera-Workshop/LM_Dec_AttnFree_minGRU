# LM_Dec_AttnFree_minGRU

minGRU - AttnFree family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (2 seed-repeat runs, see results)**
- Failure class: none

## Provenance

- Source repository: (new) lucidrains/minGRU-pytorch 0.2.1
- Upstream / license: lucidrains/minGRU-pytorch (MIT), pip 0.2.1

## Differences from the source ([Unified] protocol)

- minLM defaults (expansion 1.5, ff_mult 4, no conv)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.94 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.637 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.580 / test 1.584 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.471 / test 1.486 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 593 MB, inference 130 MB
- Peak GPU memory at length 8192 (batch 1): train 2208 MB, inference 410 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.02 (keys removed 0.01); 4 pairs @ 2048: 0.03 (keys removed 0.01); 16 pairs @ 512: 0.02 (keys removed 0.01); 16 pairs @ 2048: 0.02 (keys removed 0.02)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.02 (keys removed 0.01); 4 pairs @ 2048: 0.03 (keys removed 0.02); 4 pairs @ 8192: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.07 (keys removed 0.07); 16 pairs @ 8192: 0.07 (keys removed 0.07)
- LM-06 memory update `lm06`: @512: latest 0.01, unchanged 0.01, stale 0.02; @2048: latest 0.02, unchanged 0.01, stale 0.01

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy minGRU-pytorch==0.2.1
python run.py check mingru
python run.py lm02 mingru      # enwik8 is downloaded to data/ on first use
```
