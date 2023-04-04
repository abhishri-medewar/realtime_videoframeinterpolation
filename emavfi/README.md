# Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation 

[EMA-VFI Paper](https://arxiv.org/abs/2303.00440)

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --data_root <path to vimeo90k mos data> --model_path <path to emavfi pretrained model> 
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --data_root <path to xiph mos data> --model_path <path to emavfi pretrained model> 
```

## References

1. Zhang, Guozhen, et al. “Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation.” ArXiv:2303.00440 [Cs], 4 Mar. 2023, arxiv.org/abs/2303.00440. Accessed 4 Apr. 2023.
