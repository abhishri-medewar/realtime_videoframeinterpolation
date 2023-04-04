# Channel Attention Is All You Need for Video Frame Interpolation

[CAIN Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6693)

## Inference

1. Run the following script to generate result for Vimeo90k MOS Data.

``` 
python get_results_vimeo90k_mos.py --data_root <path to vimeo90k mos data> --model_path <path to cain pretrained model> 
```
 
2. Run the following script to generate result for Xiph MOS Data.

``` 
python get_results_xiph_mos.py --data_root <path to xiph mos data> --model_path <path to cain pretrained model> 
```

## References

1. Choi, Myungsub, et al. “Channel Attention Is All You Need for Video Frame Interpolation.” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 07, 3 Apr. 2020, pp. 10663–10671, https://doi.org/10.1609/aaai.v34i07.6693.


