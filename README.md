# [CVPR 2025] OralXrays-9: Towards Hospital-Scale Panoramic X-ray Anomaly Detection via Personalized Multi-Object Query-Aware Mining

> Authors: Bingzhi Chen, Sisi Fu, Xiaocheng Fang, Jieyi Cai, Boya Zhang, Minhua Lu, Yishu Liu.

### Abstract:

In clinical practice, panoramic dental radiography is a widely employed imaging technique that can provide a detailed and comprehensive view of dental structures and surrounding tissues for identifying various oral anomalies. However, due to the complexity of oral anomalies and the scarcity of available data, existing research still suffers from substantial challenges in automated oral anomaly detection. To this end, this paper presents a new hospital-scale panoramic X-ray benchmark, namely “OralXrays-9”, which consists of 12,688 panoramic X-ray images with 84,113 meticulously annotated instances across nine common oral anomalies. Correspondingly, we propose a personalized Multi-Object Query-Aware Mining (MOQAM) paradigm, which jointly incorporates the Distribution-IoU Region Proposal Network (DI-RPN) and Class-Balanced Spherical Contrastive Regularization (CB-SCR) mechanisms to address the challenges posed by multi-scale variations and class-imbalanced distributions. To the best of our knowledge, this is the first attempt to develop AI-driven diagnostic systems specifically designed for multi-object oral anomaly detection, utilizing publicly available data resources. Extensive experiments on the newly-published OralXrays-9 dataset and real-world nature scenarios consistently demonstrate the superiority of our MOQAM in revolutionizing oral healthcare practices.

<img src="./framework.png" width="800">

## 📚 Dataset
1. You can download our dataset using the following link:
### Download OralXrays-9 Dataset:
```sh
https://drive.google.com/drive/folders/1_y7ERcFicnOYY2DMR6Qe1W4KGdCsoQ1n?usp=drive_link
```
```sh
unzipped password: CVPR2024-OralXrays-9
```
2. The OralXrays-9 dataset should be organized as:
```
data
 ├── coco
      ├── annotations
      │    ├── instances_train2017.json
      │    └── instances_val2017.json
      ├── train2017
      └── val2017
```

## ⚙️ Installation
We implement MOQAM using `MMDetection V2.25.3` and `MMCV V1.7.0`. We test our models under requires `python=3.7.1, torch=1.11.0, torchvision=0.12.0`. 

## 🚀 Training and Testing

### Download Model Checkpoints:
```sh
https://drive.google.com/drive/folders/1Nm1673tPy_t8x69e5tvQTan0dQsGf-OM?usp=drive_link
```

### Training:
```sh
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/MOQAM.py 
```

### Testing:
```sh
CUDA_VISIBLE_DEVICES=0 python tools/test.py --model-config /path/to/model.py --checkpoint /path/to/checkpoint.pth 
```

## 📝 Citation

```sh
@inproceedings{chen2025oralxrays,
  title={OralXrays-9: Towards Hospital-Scale Panoramic X-ray Anomaly Detection via Personalized Multi-Object Query-Aware Mining},
  author={Chen, Bingzhi and Fu, Sisi and Fang, Xiaocheng and Cai, Jieyi and Zhang, Boya and Lu, Minhua and Liu, Yishu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15570--15579},
  year={2025}
}
```

If you have any questions, please get in touch with us: chenbingzhi@bit.edu.cn or fangxiaocheng162@gmail.com.
