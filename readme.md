# About
This is a more feature complete python script for interacting with a ONNX converted version of Stable Diffusion on a Windows system. To get setup, start by following the [Stable Diffusion for AMD GPUs on Windows using DirectML](https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269) instructions created by [harishhanand95](https://gist.github.com/harishanand95). I feel they are the primary person who's gotten Stable Diffusion working on non NVIDIA hardware. Be sure to thank them. Instead of following the **Run Stable Diffusion on AMD GPUs** steps, run `onnx_txt2img.py` instead.

My script file is inspired by Travel Neil's post at [https://www.travelneil.com/stable-diffusion-windows-amd.html](https://www.travelneil.com/stable-diffusion-windows-amd.html). I took Travel Neil's script and added the CLI arguments found in [Stable Diffusion's](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py) `txt2img.py` script. Extra argument I added include the option to run Stable Diffusion ONNX on a GPU through DirectML or even on a CPU. I've also included an option to generate a random seed value.

# Example Command
```
python .\onnx_txt2img.py --n_samples 1 --prompt "astronaut riding a horse" --random_seed True
100%|██████████████████████████████████████████████████████████████████████████████████| 51/51 [02:03<00:00,  2.42s/it]
seed: 391198996415909456
```
![astronaut riding a horse](/docs/astronaut_riding_a_horse.png)

By default the script will attempt to generate 3 images at 512x512 pixels in a `outputs\txt2img-samples\samples` directory, the same as Stable Diffusion's txt2img.py script. This can be taxing even on a system with 16GB of VRAM. I recommend turning the sample size down to 1-2 or reducing the output resolution.

# All arguement options
```
onnx_txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--ddim_steps DDIM_STEPS] [--ddim_eta DDIM_ETA] [--H H] [--W W] [--n_samples N_SAMPLES] [--scale SCALE] [--seed SEED] [--random_seed RANDOM_SEED] [--hardware {gpu,cpu}] [--loop LOOP] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     The prompt to render
  --outdir [OUTDIR]     Default: outputs/txt2img-samples. Directory to write results to
  --ddim_steps DDIM_STEPS
                        Default: 50. Number of ddim sampling steps
  --ddim_eta DDIM_ETA   Default: 0.0. ddim eta (eta=0.0 corresponds to deterministic sampling)
  --H H                 Default: 512. Image height, in pixel space
  --W W                 Default: 512. Image width, in pixel space
  --n_samples N_SAMPLES
                        Default: 2. How many samples to produce for each given prompt. A.k.a. batch size
  --scale SCALE         Default: 7.5. Unconditional guidance scale
  --seed SEED           Default: 42. The seed (for reproducible sampling)
  --random_seed RANDOM_SEED
                        Default: False. Generate a random seed value
  --hardware {gpu,cpu}  Default: gpu. GPU or CPU processing
  --loop LOOP           Default: 1. How many times to loop through. USE WITH --random_seed flag
  --log LOG             Default: False. Create a logfile detailing the image parameters
  ```