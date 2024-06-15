# VFE_TPS
Code for our paper "Enhancing Visual Representation for Text-based Person Searching"



## Usage
### Requirements

```
PyTorch 2.0.0
torchvision 0.15.0
easydict
tqdm
prettytable
```
 ### Prepare Datasets
 The CUHK-PEDES dataset is proposed by the paper "Person Search with Natural Language Description" (Shuang Li, et al.) and can be downloaded from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description). ICFG-PEDES dataset can be found [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset is [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).
 ```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

### Training
Directly run "run.sh" file 

### Testing
```
python test.py
```

## Results
We use single Nvidia RTX 3090 GPU (24G) for training and testing
### Results on CUHK-PEDES dataset
|Method|Rank-1|Rank-5|Rank-10|mAP|
|--|--|--|--|--|
|AXM-Net|64.44|80.52|86.77|58.73|
|LGUR|64.21|81.94|87.93|-|
|IVT|65.59|83.11|89.21|-|
|CFine|69.57|85.93|91.15|-|
|TP-TPS|70.16|86.10|90.98|66.32|
|VGSG|71.38|86.75|91.86|-|
|VFE-TPS (ours)|72.47|88.24|93.24|64.26|

### Results on ICFG-PEDES dataset
|Method|Rank-1|Rank-5|Rank-10|mAP|
|--|--|--|--|--|
|LGUR|59.02|75.32|81.56|-|
|MANet|59.44|76.80|82.75|-|
|CFine|60.83|76.55|82.42|-|
|TP-TPS|60.64|75.97|81.76|42.78|
|VFE-TPS (ours)|62.71|78.73|84.51|43.08|

### Results on RSTPReid dataset
|Method|Rank-1|Rank-5|Rank-10|mAP|
|--|--|--|--|--|
|LBUL|45.55|68.20|77.85|-|
|IVT|46.70|70.00|78.80|-|
|CFine|50.55|72.50|81.60|-|
|TP-TPS|50.65|72.45|81.20|43.11|
|VFE-TPS (ours)|59.25|81.90|88.85|45.96|

## Acknowledgements
Our code is partially based on [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA). Sincerely appreciate for their contributions.

