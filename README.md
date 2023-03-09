```bash
conda create -n stable-diffusion
conda activate stable-diffusion
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install diffusers["torch"]
pip install diffusers==0.12.1
mkdir images
```

```bash
python txt2img.py
```