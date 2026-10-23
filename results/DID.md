# Difference-in-differences sweep

`uv run causal-uplift did --units 2000 --reps 200` (seed 20260830; true effect 0.05; selection 1.0).

```
 trend_gap       estimator  mean_estimate  mean_bias  mc_se_bias  coverage
    0.0000 post_difference         0.2164     0.1664      0.0007    0.0000
    0.0000        pre_post         0.1506     0.1006      0.0003    0.0000
    0.0000         did_2x2         0.0502     0.0002      0.0004    0.9600
    0.0100 post_difference         0.2248     0.1748      0.0006    0.0000
    0.0100        pre_post         0.1600     0.1100      0.0003    0.0000
    0.0100         did_2x2         0.0604     0.0104      0.0004    0.6150
    0.0300 post_difference         0.2457     0.1957      0.0007    0.0000
    0.0300        pre_post         0.1803     0.1303      0.0003    0.0000
    0.0300         did_2x2         0.0799     0.0299      0.0005    0.0150
    0.0500 post_difference         0.2657     0.2157      0.0007    0.0000
    0.0500        pre_post         0.1998     0.1498      0.0003    0.0000
    0.0500         did_2x2         0.1000     0.0500      0.0004    0.0000
```
