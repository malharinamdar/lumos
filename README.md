# Lumos — Stable Diffusion in PyTorch

PyTorch implementation and exploration of diffusion-based generative models inspired by:

> *Denoising Diffusion Probabilistic Models (DDPM)*

The project focuses on understanding and reproducing the core architectural components behind Stable Diffusion, including latent diffusion, UNet denoising, CLIP conditioning, and VAE-based image representations.

---

## Features

- Stable Diffusion inference pipeline
- DDPM-based diffusion implementation
- CLIP text encoder integration
- VAE encoder and decoder modules
- Attention mechanisms and latent-space generation
- Prompt-conditioned image synthesis in PyTorch

---

## Repository Structure

```text
lumos/
├── data/
├── images/
├── stable_diffusion/
│   ├── attention.py
│   ├── clip.py
│   ├── ddpm.py
│   ├── diffusion.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── pipeline.py
│   ├── model_loader.py
│   ├── model_converter.py
│   ├── evaluate.py
│   └── demo.ipynb
├── requirements.txt
├── Stable_Diffusion_Diagrams_V2.pdf
└── README.md
```

---

## Setup

Download:

`v1-5-pruned-emaonly.ckpt`

from:

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

and place it inside:

```text
data/
```

---

## Demonstration

Example inference and image generation workflows are available in:

```text
stable_diffusion/demo.ipynb
```

---

## References

- *Denoising Diffusion Probabilistic Models* — Ho et al., 2020
- https://medium.com/@ninads79shukla/building-a-stable-diffusion-model-from-scratch-with-pytorch-part-1-b358106a846b
- https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946
