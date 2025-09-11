# Model Zoo

This repository includes our trained models for all experiments. We also provide performance results and pretrained weight files.

## Benchmark of Multi-Source Domain Generalization

The following two tables show the benchmarks on Diverse Weather and Real to Artistic, respectively. *`Src.`* denotes the performance on the source domain. *`Tgt.`* denotes the performance on the target domain. It's the main metric to be optimized in multi-source domain generalization for object detection.

### Diverse Weather (mAP50)

We adopt five datasets to assess generalization on the real-to-artistic domain gap: Passcal VOC (VOC), MS COCO (COCO), Clipart (CP), Comic (CM), and Watercolor (WC). We use VOC+COCO as the (real-world) multi-source domains, and CP, CM, and WC as the (art-style) target domains.

|                     Method                   |  DS (*Src.1*)  |  NC (*Src.2*)  |  DF (*Tgt.1*)  |  DR (*Tgt.2*)  |  NR (*Tgt.3*)  |    download    |
|----------------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|
| [CLIP-Gap](https://arxiv.org/abs/2301.05499) |      50.0      |      51.2      |      34.0      |      28.6      |      20.2      |                |
| [PDOC](https://arxiv.org/abs/2402.18447)     |      52.3      |      50.6      |    **38.3**    |      30.5      |      23.4      |                |
| [OADG](https://arxiv.org/abs/2312.12133)     |      44.7      |      35.6      |      30.6      |      25.2      |      18.0      |                |
|  MS-DePro (ours)                             |    **55.5**    |    **62.8**    |    **38.3**    |    **35.2**    |    **29.7**    | [model]() \| [metrics]() |

### Real to Artistic (mAP50)

We adopt five datasets covering different weather conditions: Daytime Sunny (DS), Night Clear (NC), Daytime Foggy (DF), Dusk Rainy (DR), and Night Rainy (NR). We use DS+NC as the multi-source domains, and DF, DR, and NR as the target domains. Note that MS COCO is not evaluated as it is only used for multi-source training (See Tab. 2 in the paper for details).

|                     Method                   |  VOC (*Src.1*)  |  CP (*Tgt.1*)  |  CM (*Tgt.2*)  |  WC (*Tgt.3*)  |    download    |
|----------------------------------------------|-----------------|----------------|----------------|----------------|----------------|
| [CLIP-Gap](https://arxiv.org/abs/2301.05499) |       -         |       -        |      32.7      |      48.1      |                |
| [PDOC](https://arxiv.org/abs/2402.18447)     |      76.4       |      16.7      |      27.2      |      41.7      |                |
| [OADG](https://arxiv.org/abs/2312.12133)     |      78.7       |      29.7      |      20.4      |      44.0      |                |
|  MS-DePro (ours)                             |    **78.8**     |    **47.5**    |    **42.5**    |    **57.5**    | [model]() \| [metrics]() |

## Benchmark of Multi-Source Domain Adaptation

The following three tables show the benchmarks on Cross-time, Cross-camera, and Mixed-domain, respectively. We show two baseline settings for comparison: UDA, MSDA, and Oracle. (1) UDA: all source domains are combined into a single set. (2) MSDA: all MSDA methods designed for object detection.

`A+B` indicates that `A` and `B` are used as multi-source domains. We evaluate on the Dawn/Dusk of BDD100K for Cross-time, and Daytime of BDD100K for Cross-camera and Mixed-domain.

| Setting |                        Method                      |  Cross-time (D+N)  |  Cross-camera (C+K)  |  Mixed-domain (C+M)  |  Mixed-domain (C+M+S)  |
|---------|----------------------------------------------------|--------------------|----------------------|----------------------|------------------------|
|   UDA   | [UMT](https://arxiv.org/abs/2003.00707)            |        33.5        |         47.0         |         18.5         |          25.1          |
|         | [Adapt. Teacher](https://arxiv.org/abs/2111.13216) |        34.6        |         48.4         |         22.9         |          29.6          |
|   MSDA  | [DMSN](https://arxiv.org/abs/2106.15793)           |        35.0        |         49.2         |          -           |           -            |
|         | [TPKP](https://arxiv.org/abs/2204.07964)           |        39.8        |         58.4         |         35.3         |          37.1          |
|         | [PMT](https://arxiv.org/abs/2309.14950)            |        45.3        |         58.7         |         38.7         |          39.7          |
|         | [ACIA](https://arxiv.org/abs/2403.09918)           |        47.9        |         59.1         |         41.0         |          42.3          |
|         | MS-DePro (ours)                                    |      **53.7**      |       **68.4**       |       **43.9**       |        **46.7**        |

| download |        Cross-time        |       Cross-camera       |    Mixed-domain (C+M)    |   Mixed-domain (C+M+S)   |
|----------|--------------------------|--------------------------|--------------------------|--------------------------|
|          | [model]() \| [metrics]() | [model]() \| [metrics]() | [model]() \| [metrics]() | [model]() \| [metrics]() | 
