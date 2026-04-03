# Lightweight Feature Watermark for Robust Image Attribution in Latent Diffusion Models

### Prepare

```
pip install -r requirements.txt
```

### Dataset and Model

Download the datasets and models from Google Drive: [download](https://drive.google.com/file/d/1jBtWOJPyQBlXLjmj5FwDTDvohma0pgDY/view?usp=drive_link)

### Usage

#### 1.pretrain 

```
python vectors_generation.py
```

#### 2.train encoder and decoder

```
python train.py
```

#### 3.test watermark

```
python test.py
```
### Citation
```
@ARTICLE{11458787,
  author={Shen, Zhangyi and Zhuang, Yuzhong and Zhu, Yani and Yao, Ye},
  journal={IEEE Signal Processing Letters}, 
  title={Lightweight Feature Watermark for Robust Image Attribution in Latent Diffusion Models}, 
  year={2026},
  pages={1-5},
  keywords={Payloads;Feeds;Internet of Things;Digital images;Internet;Cyberspace;Communication systems;Communication networks;Computer networks;Codecs;Latent Diffusion Model;Robust watermark;Image attribution;Lightweight},
  doi={10.1109/LSP.2026.3679279}}
```


