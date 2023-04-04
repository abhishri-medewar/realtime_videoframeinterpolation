# Real-Time Intermediate Flow Estimation for Video Frame Interpolation

[Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294)

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --data_root <path to vimeo90k mos data> --model <path to rife pretrained model> 
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --data_root <path to xiph mos data> --model <path to rife pretrained model> 
```

## References

1. Huang, Zhewei, et al. “Real-Time Intermediate Flow Estimation for Video Frame Interpolation.” ArXiv:2011.06294 [Cs], 13 July 2022, arxiv.org/abs/2011.06294. Accessed 4 Apr. 2023.
