# FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation

[Flavr paper](https://arxiv.org/pdf/2012.08512.pdf)

FLAVR is a fast, flow-free frame interpolation method capable of single shot multi-frame prediction. It uses a customized encoder decoder architecture with spatio-temporal convolutions and channel gating to capture and interpolate complex motion trajectories between frames to generate realistic high frame rate videos. This repository contains original source code.

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --data_root <path to vimeo90k mos data> --model_path <path to flavr pretrained model> 
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --data_root <path to xiph mos data> --model_path <path to flavr pretrained model> 
```
 

## References

1. Kalluri, Tarun, et al. “FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation.” 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2023, pp. 2070–81. DOI.org (Crossref), https://doi.org/10.1109/WACV56688.2023.00211.
