# AnyText2: Visual Text Generation and Editing With Customizable Attributes

<a href='https://arxiv.org/abs/2411.15245'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/tyxsspa/AnyText2'><img src='https://img.shields.io/badge/Code-Github-green'></a> <a href='https://modelscope.cn/studios/iic/studio_anytext2'><img src='https://img.shields.io/badge/Demo-ModelScope-lightblue'></a> <a href='https://huggingface.co/spaces/modelscope/AnyText2'><img src='https://img.shields.io/badge/Demo-HuggingFace(TODO)-yellow'></a>

![sample](docs/sample.jpg "sample")

## 📌News
[2025.03.01] - 🧨We release all code, checkpoint and dataset of AnyText2, and the online demo is available on [ModelScope](https://modelscope.cn/studios/iic/studio_anytext2)!  
[2024.11.22] - The paper is available at [arxiv](https://arxiv.org/abs/2411.15245).  


## ⏰TODOs

- [ ] Release SDXL version of AnyText2 (AnyText2XL)
- [x] Release inference/training/evaluation code, checkpoint, dataset of AnyText2
 

## 💡Methodology
AnyText2 is a novel method that enables precise control over multilingual text attributes in natural scene image generation and editing. Our approach consists of two main components. First, we propose a WriteNet+AttnX architecture that injects text rendering capabilities into a pre-trained T2I model.  Compared to its predecessor, AnyText, our new approach not only enhances image realism but also achieves a 19.8% increase in inference speed.  Second, we explore techniques for extracting fonts and colors from scene images and develop a Text Embedding Module that encodes these text attributes separately as conditions. As an extension of AnyText, this method allows for customization of attributes for each line of text, leading to improvements of 3.3% and 9.3% in text accuracy for Chinese and English, respectively. Through comprehensive experiments, we demonstrate the state-of-the-art performance of our method.

![framework](docs/framework.jpg "framework")

## 🛠Installation
```bash
# Download checkpoint (pip install modelscope first)
from modelscope import snapshot_download
your_path_to_model_dir = snapshot_download('iic/cv_anytext2')
# Clone anytext code and move checkpoint
git clone https://github.com/tyxsspa/AnyText2.git
cd AnyText2 && mkdir -p models  &&  sleep 1 && mv your_path_to_model_dir/* models
# Create a new environment and install packages as follows:
conda env create -f environment.yaml
conda activate anytext2
```

## 🔮Inference

**[Recommend]**： We release a demo on [ModelScope](https://modelscope.cn/studios/iic/studio_anytext2)

If you have advanced GPU (with at least 8G memory), it is recommended to deploy our demo as below, which includes usage instruction, user interface and abundant examples.
```bash
python demo.py
```
The other parameters of run the demo are consistent with [AnyText](https://github.com/tyxsspa/AnyText?tab=readme-ov-file#inference).

## 📈Evaluation

The evaluation scripts are located in `AnyText2/eval`, and their usage remains consistent with [AnyText](https://github.com/tyxsspa/AnyText?tab=readme-ov-file#evaluation). We have also updated the evaluation files for the long caption version in [AnyText-benchmark](https://modelscope.cn/datasets/iic/AnyText-benchmark/summary).

We trained the **AnyText2XL** model based on diffusers, using the SDXL model and 1024x1024 version of AnyWord-3M dataset, it will be released soon.
![eval](docs/eval.jpg "eval")


## 🚂Training

We have updated the training dataset [AnyWord-3M](https://modelscope.cn/datasets/iic/AnyWord-3M), where the image data remains the same, only annotation files for training AnyText2 are added. These files include long captions and text color labels. You can start the training by running `python train.py`, and the details are consistent with the documentation for [AnyText](https://github.com/tyxsspa/AnyText?tab=readme-ov-file#training).


## Citation
```
@article{tuo2023anytext,
      title={AnyText: Multilingual Visual Text Generation And Editing}, 
      author={Yuxiang Tuo and Wangmeng Xiang and Jun-Yan He and Yifeng Geng and Xuansong Xie},
      year={2023},
      eprint={2311.03054},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{tuo2024anytext2,
      title={AnyText2: Visual Text Generation and Editing With Customizable Attributes}, 
      author={Yuxiang Tuo and Yifeng Geng and Liefeng Bo},
      year={2024},
      eprint={2411.15245},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
