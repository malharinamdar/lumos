# Lumos: Stable Diffusion inference in PyTorch

A from-scratch reproduction of **Stable Diffusion v1.5 inference** in plain PyTorch, built to understand how latent diffusion works end to end. It loads the official v1.5 weights into modules written directly in PyTorch, without the diffusers library (VAE, CLIP text encoder, U-Net and DDPM sampler), and generates 512 × 512 images from a text prompt or from an input image.

> **Credit.** This code follows Umar Jamil's *Coding Stable Diffusion from scratch in PyTorch*
> ([video](https://www.youtube.com/watch?v=ZBKpAp_6TGI), [repository](https://github.com/hkproj/pytorch-stable-diffusion)).
> The model and pipeline files are based on his implementation, which is released under the MIT License (see [LICENSE](LICENSE)).

## How it works

1. **Text encoding.** The prompt is tokenized with the CLIP tokenizer and encoded by the CLIP text encoder into a 77 × 768 sequence of embeddings ([`clip.py`](<stable diffusion/clip.py>)).
2. **Starting latent.** Generation starts from random noise in a 4 × 64 × 64 latent space. For image-to-image, the input image is first encoded by the VAE encoder and partly noised, controlled by `strength` ([`encoder.py`](<stable diffusion/encoder.py>)).
3. **Denoising.** At each step the U-Net predicts the noise in the latent, attending to the text embeddings through cross-attention ([`diffusion.py`](<stable diffusion/diffusion.py>), [`attention.py`](<stable diffusion/attention.py>)). With classifier-free guidance, it also makes a prediction for an empty prompt and pushes the result away from it by `cfg_scale`.
4. **Sampling.** The DDPM sampler removes the predicted noise over `n_inference_steps` steps ([`ddpm.py`](<stable diffusion/ddpm.py>)).
5. **Decoding.** The VAE decoder turns the final latent back into a 512 × 512 image ([`decoder.py`](<stable diffusion/decoder.py>)).

## Files

| File | What it does |
|---|---|
| `stable diffusion/clip.py` | CLIP text encoder |
| `stable diffusion/encoder.py`, `decoder.py` | VAE encoder and decoder |
| `stable diffusion/diffusion.py` | U-Net denoiser with time embeddings |
| `stable diffusion/attention.py` | Self-attention and cross-attention |
| `stable diffusion/ddpm.py` | DDPM noise schedule and sampler |
| `stable diffusion/pipeline.py` | The generation loop (text-to-image and image-to-image, classifier-free guidance) |
| `stable diffusion/model_converter.py`, `model_loader.py` | Map the original v1.5 checkpoint onto these modules and load them |
| `stable diffusion/demo.ipynb` | Example run |
| `stable diffusion/add_noise.ipynb` | Visualises the forward (noising) process |
| `data/` | CLIP tokenizer files; the model checkpoint goes here |

## Run

```bash
pip install -r requirements.txt
```

1. Download `v1-5-pruned-emaonly.ckpt` from [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and put it in `data/`. The tokenizer files are already there.
2. Open `stable diffusion/demo.ipynb`, set the prompt, and run it. Set `ALLOW_CUDA` or `ALLOW_MPS` to `True` to use a GPU.

## References

- J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022.
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
- J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance*. 2022.
- U. Jamil. [*Coding Stable Diffusion from scratch in PyTorch*](https://www.youtube.com/watch?v=ZBKpAp_6TGI) and [hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion).
