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

