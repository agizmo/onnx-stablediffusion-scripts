# About
This is a more feature-complete python script for interacting with an ONNX converted version of Stable Diffusion on a Windows or Linux system. Original instructions were created by[harishhanand95](https://gist.github.com/harishanand95) and are available at [Stable Diffusion for AMD GPUs on Windows using DirectML](https://gist.github.com/harishanand95/75f4515e6187a6aa3261af6ac6f61269). I feel they are the primary person who's gotten Stable Diffusion working on non-NVIDIA hardware. Be sure to thank them.

My scripts were inspired by Travel Neil's post at [https://www.travelneil.com/stable-diffusion-windows-amd.html](https://www.travelneil.com/stable-diffusion-windows-amd.html). I took Travel Neil's script and added the CLI arguments found in [Stable Diffusion's](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py) `txt2img.py` script. Extra arguments I added include the option to run Stable Diffusion ONNX on a GPU through DirectML or even on a CPU. I've also included an option to generate a random seed value.

The setup has been simplified thanks to a guide by [averad](https://gist.github.com/averad). Their guide is available at [Stable Diffusion for AMD GPUs on Windows using DirectML](https://gist.github.com/averad/256c507baa3dcc9464203dc14610d674) and has been updated with a lot of great info, including how to use customized weight checkpoint files with ONNX.

# Setup
1. Install Git.
    1. [https://git-scm.com/](https://git-scm.com/)
1. Install Miniconda.
    1. [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
1. Launch Conda.
    1. Windows will have a new icon in the start menu labeled `Anaconda Prompt (miniconda3)`
1. Download this project from GitHub.
    1. `git clone https://github.com/agizmo/onnxstablediffusionscripts.git`
1. Create a new Conda environment using the environment files for your given OS.
    1. `conda env create --file environment.yml`
1. Activate the new Conda environment.
    1. `conda activate onnx`
1. (Windows Only) Force install onnxruntime-directml. ¯\\_(ツ)_/¯ I Don't know why force install is required, but it worked for averad.
    1. `pip install onnxruntime-direcml --force-reinstall`  
    **OR**
    1. Install the latest 1.14 nightly build of ONNX. The nightlys appear to have significant performance improvements over the released versions available through pip. (I've seen a 2-3x speed increase)
        1. Go to [https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/versions/](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/versions/)
        1. Click on the latest version available. Download the WHL file for your Python environment. If you used the environment file above to set up Conda, choose the `cp39` file (aka Python 3.9).
        1. Run the command `pip install "path to the downloaded WHL file" --force-reinstall` to install the package.
1. Download the weights for Stable Diffusion. Currently, the best options are to choose between versions 1.4, 1.5, and 2.0. Choose one below:
    1. In a web browser go to [https://huggingface.co](https://huggingface.co) and create an account.
    1. For Stable Diffusion 1.4:
        1. Go to [https://huggingface.co/CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
        1. Read the license details. By clicking Accept, you agree to share your Hugging Face contact information with the developers.
        1. Go back to the command prompt to download the Stable Diffusion weights.
            1. Download the repo with the weights. You should be prompted to enter your Hugging Face username and password the first time. The clone job will take several minutes to process and download. Be patient.  
            `git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 --branch onnx --single-branch stable_diffusion_onnx`
    1. Stable Diffusion 1.5
        1. Go to [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
        1. Read the license details. By clicking Accept, you agree to share your Hugging Face contact information with the developers.
        1. Go back to the command prompt to download the Stable Diffusion weights. 
            1. Download the repo with the weights. You should be prompted to enter your Hugging Face username and password the first time. The clone job will take several minutes to process and download. Be patient.  
            `git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 --branch onnx --single-branch stable_diffusion_onnx`
    1. Stable Diffusion 2.0  
    Currently, there is no prebuilt ONNX version of SD 2.0. The current weights must be converted to work.
        1. Download Hugging Face's script to convert Stable Diffusion models to ONNX.
            1. Windows  
            `powershell -Command Invoke-WebRequest -Uri "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py" -OutFile convert_stable_diffusion_checkpoint_to_onnx.py`
            1. Linux  
            `wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py`
        1. Use the ONNX conversion script to download Stable Diffusion's 2.0 model. The job will take several minutes to process and download. Be patient.  
        `python convert_stable_diffusion_checkpoint_to_onnx.py --model_path "stabilityai/stable-diffusion-2" --output_path "stable_diffusion_onnx"`

1. **Setup Complete**

# Using
Each time you launch Conda, you need to activate the environment set up earlier and change to the directory where you downloaded this repository.
1. Launch Conda.
    1. Windows will have a new icon in the start menu labeled `Anaconda Prompt (miniconda3)`
1. Activate the Conda environment setup before.
    1. `conda activate onnx`
1. Change to the correct directory.
    1. `cd "path to onnx-stablediffusion-scripts folder"`
1. Run the scripts with the desired parameters.

# Example Commands
```
python onnx_txt2img.py --prompt "astronaut riding a horse" --random_seed True
seed: 3844704755
100%|██████████████████████████████████████████████████████████████████████████████████| 51/51 [00:26<00:00,  1.90it/s]
```
![astronaut riding a horse](/docs/astronaut_riding_a_horse.png)

By default, the script will attempt to generate 1 image at 512x512 pixels in a `outputs\txt2img-samples\samples` directory, the same as Stable Diffusion's txt2img.py script. This can be taxing even on a system with 8GB of VRAM.


```
python onnx_img2img.py --prompt "cowboy riding a pig" --init_img ".\outputs\txt2img-samples\samples\00001.png" --random_seed true
loaded input image of size (512, 512) from .\outputs\txt2img-samples\samples\00292.png
seed: 2912103985
100%|█████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:24<00:00,  2.01it/s]
```
![cowboy riding a pig](/docs/cowboy_riding_a_pig.png)

By default the script will attempt to generate 1 image at 512x512 pixels in a `outputs\img2img-samples\samples` directory, the same as Stable Diffusion's img2img.py script. This can be taxing even on a system with 8GB of VRAM.

# All arguement options
## Text to Image Generation
```
onnx_txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--ddim_steps DDIM_STEPS] [--ddim_eta DDIM_ETA] [--H H] [--W W] [--n_samples N_SAMPLES] [--scale SCALE] [--seed SEED] [--random_seed RANDOM_SEED] [--negative_prompt NEGATIVE_PROMPT] [--hardware {directml,cpu}] [--loop LOOP] [--log LOG]

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
                        Default: 1. How many samples to produce for each given prompt. A.k.a. batch size
  --scale SCALE         Default: 7.5. Unconditional guidance scale
  --seed SEED           Default: 42. The seed (for reproducible sampling)
  --random_seed RANDOM_SEED
                        Default: False. Generate a random seed value
  --negative_prompt NEGATIVE_PROMPT
                        Default: ''. Prompts to not guide the image generation
  --hardware {directml,cpu}
                        Default: directml. DirectML or CPU processing
  --loop LOOP           Default: 1. How many times to loop through. USE WITH --random_seed flag
  --log LOG             Default: False. Create a logfile detailing the image parameters. CSV file is saved in outdir
  ```
## Image to Image Generation
  ```
onnx_img2img.py [-h] [--prompt [PROMPT]] [--init_img [INIT_IMG]] [--outdir [OUTDIR]] [--ddim_steps DDIM_STEPS] [--ddim_eta DDIM_ETA] [--H H] [--W W] [--n_samples N_SAMPLES] [--scale SCALE][--model MODEL] [--seed SEED] [--random_seed RANDOM_SEED] [--negative_prompt NEGATIVE_PROMPT] [--hardware {directml,cpu}] [--loop LOOP] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     The prompt to render
  --init_img [INIT_IMG]
                        path to the input image
  --outdir [OUTDIR]     Default: outputs/img2img-samples. Directory to write results to
  --ddim_steps DDIM_STEPS
                        Default: 50. Number of ddim sampling steps
  --ddim_eta DDIM_ETA   Default: 0.0. ddim eta (eta=0.0 corresponds to deterministic sampling)
  --H H                 Default: 512. Image height, in pixel space
  --W W                 Default: 512. Image width, in pixel space
  --n_samples N_SAMPLES
                        Default: 1. How many samples to produce for each given prompt. A.k.a. batch size
  --scale SCALE         Default: 7.5. Unconditional guidance scale
  --model MODEL         Default: ./stable_diffusion_onnx. Path to the model folder
  --seed SEED           Default: 42. The seed (for reproducible sampling)
  --random_seed RANDOM_SEED
                        Default: False. Generate a random seed value
  --negative_prompt NEGATIVE_PROMPT
                        Default: ''. Prompts to not guide the image generation
  --hardware {directml,cpu}
                        Default: directml. DirectML or CPU processing
  --loop LOOP           Default: 1. How many times to loop through. USE WITH --random_seed flag
  --log LOG             Default: False. Create a logfile detailing the image parameters. CSV file is saved in outdir
  ```