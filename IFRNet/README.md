# IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

[IFRNet Paper](https://arxiv.org/abs/2205.14620)

Existing flow-based frame interpolation methods almost all first estimate or model intermediate optical flow, and then use flow warped context features to synthesize target frame. However, they ignore the mutual promotion of intermediate optical flow and intermediate context feature. Also, their cascaded architecture can substantially increase the inference delay and model parameters, blocking them from lots of mobile and real-time applications.A single encoder-decoder based IFRNet for compactness and fast inference.

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --data_root <path to vimeo90k mos data> --model_path <path to ifrnet pretrained model> 
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --data_root <path to xiph mos data> --model_path <path to ifrnet pretrained model> 
```

## References

1. Kong, Lingtong, et al. “IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation.” ArXiv:2205.14620 [Cs], 29 May 2022, arxiv.org/abs/2205.14620. Accessed 4 Apr. 2023.
