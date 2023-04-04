# FILM: Frame Interpolation for Large Motion

A unified single-network approach that doesn't use additional pre-trained networks, like optical flow or depth, and yet achieve state-of-the-art results. Use a multi-scale feature extractor that shares the same convolution weights across the scales. Model is trainable from frame triplets alone. <br>

[FILM: Frame Interpolation for Large Motion](https://arxiv.org/abs/2202.04901)

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --input_data_path <path to vimeo90k mos data> --model_path <path to film trained model> \
--output_path <path to save results>
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --input_data_path <path to xiph mos data> --model_path <path to film trained model> \
--output_path <path to save results>
```
 
## References

1. Reda, Fitsum, et al. “FILM: Frame Interpolation for Large Motion.” ArXiv:2202.04901 [Cs], 16 July 2022, arxiv.org/abs/2202.04901. Accessed 4 Apr. 2023.

