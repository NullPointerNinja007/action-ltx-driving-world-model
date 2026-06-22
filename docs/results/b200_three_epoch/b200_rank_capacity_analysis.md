# B200 V4 Rank/Capacity Campaign Analysis

This run tests whether V4 main text was limited by rank/capacity or simply needed longer training.

## Latest Checkpoint Summary

| method | step | rank | FVD-style | PSNR | SSIM | sharpness | motion | FFT HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v4_main_text_r32_3epoch | 23976 | 32 | 89.23053152507566 | 17.25652648138014 | 0.7002194239325865 | 0.35950030416461665 | 0.7147261832898053 | 0.9768982739429294 |
| v4_main_text_r64_3epoch | 23976 | 64 | 86.80549107376885 | 17.268756475341235 | 0.7004487027102577 | 0.3592622981954884 | 0.7107900182383851 | 0.9733018800750471 |
| v4_main_text_r128_partial | 15000 | 128 | 85.80875018111584 | 17.2757005045237 | 0.7010553973720486 | 0.35238972214465436 | 0.7074009895156268 | 0.9833078510383171 |

## Counterfactual Sensitivity

| method | step | mean RGB MAE | max RGB MAE |
|---|---:|---:|---:|
| v4_main_text_r128_partial | 6000 | 2.2740 | 3.3812 |
| v4_main_text_r128_partial | 7992 | 3.2806 | 3.7394 |
| v4_main_text_r128_partial | 10000 | 2.9760 | 3.4498 |
| v4_main_text_r128_partial | 12000 | 2.9443 | 3.2741 |
| v4_main_text_r128_partial | 15000 | 2.5914 | 2.6877 |
| v4_main_text_r32_3epoch | 7992 | 2.4368 | 3.6141 |
| v4_main_text_r32_3epoch | 10000 | 3.8927 | 3.9975 |
| v4_main_text_r32_3epoch | 15984 | 2.6369 | 3.5879 |
| v4_main_text_r32_3epoch | 18000 | 2.3013 | 2.9693 |
| v4_main_text_r32_3epoch | 20000 | 2.0884 | 2.7706 |
| v4_main_text_r32_3epoch | 22000 | 2.5688 | 3.4291 |
| v4_main_text_r32_3epoch | 23976 | 1.9036 | 2.0498 |
| v4_main_text_r64_3epoch | 7992 | 2.7354 | 3.9730 |
| v4_main_text_r64_3epoch | 10000 | 2.4542 | 3.0554 |
| v4_main_text_r64_3epoch | 15984 | 2.7628 | 3.0285 |
| v4_main_text_r64_3epoch | 18000 | 3.4990 | 3.9235 |
| v4_main_text_r64_3epoch | 22000 | 2.4576 | 3.6174 |
| v4_main_text_r64_3epoch | 23976 | 3.1566 | 3.5340 |

## Interpretation Defaults

- If rank32/rank64 improve counterfactual sensitivity without hurting FVD/motion, V4 was capacity-limited.
- If rank16 continuation improves sensitivity while rank32/rank64 do not, longer training mattered more than rank.
- If all curves stay flat, the final story should be framed as action-conditioning interference rather than successful control.
