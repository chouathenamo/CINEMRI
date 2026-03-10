---
license: cc-by-nc-sa-4.0
task_categories:
- image-segmentation
language:
- en
tags:
- medical
- image
pretty_name: 'acdc'
size_categories:
- n<1K
---


## About
This is a redistribution of the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). 
- 300 cardiac MR images and corresponding segmentation masks
- No change to any image or segmentation mask
- Files are rearranged into `Images` and `Masks` folders

This dataset is released under the `CC BY-NC-SA 4.0` license.


## News 🔥
- [10 Oct, 2025] This dataset is integrated into 🔥[MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision)🔥


## Segmentation Labels
```python
labels_map = {
    "1": "right ventricular cavity",
    "2": "myocardium",
    "3": "left ventricular cavity",
}
```


## Official Release
For more information, please go to these sites:
- Data License (official): [CC BY-NC-SA 4.0](https://humanheart-project.creatis.insa-lyon.fr/database/#item/66e290e0961576b1bad4ee3c)
- Challenge (official): https://www.creatis.insa-lyon.fr/Challenge/acdc/
- Data (official): https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html


## Download from Huggingface
```bash
#!/bin/bash
pip install huggingface-hub[cli]
huggingface-cli login --token $HF_TOKEN
```
```python
# python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="YongchengYAO/ACDC", repo_type='dataset', local_dir="/your/local/folder")
```