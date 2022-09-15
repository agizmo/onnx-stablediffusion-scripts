from diffusers import StableDiffusionOnnxPipeline
import numpy as np
import argparse, os, random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
#parser.add_argument(
#    "--skip_grid",
#    action='store_true',
#    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
#)
#parser.add_argument(
#    "--skip_save",
#    action='store_true',
#    help="do not save individual samples. For speed measurements.",
#)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
#parser.add_argument(
#    "--plms",
#    action='store_true',
#    help="use plms sampling",
#)
#parser.add_argument(
#    "--laion400m",
#    action='store_true',
#    help="uses the LAION400M model",
#)
#parser.add_argument(
#    "--fixed_code",
#    action='store_true',
#    help="if enabled, uses the same starting code across samples ",
#)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
#parser.add_argument(
#    "--n_iter",
#    type=int,
#    default=2,
#    help="sample this often",
#)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
#parser.add_argument(
#    "--C",
#    type=int,
#    default=4,
#    help="latent channels",
#)
#parser.add_argument(
#    "--f",
#    type=int,
#    default=8,
#    help="downsampling factor",
#)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
#parser.add_argument(
#    "--n_rows",
#    type=int,
#    default=0,
#    help="rows in the grid (default: n_samples)",
#)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
#parser.add_argument(
#    "--from-file",
#    type=str,
#    help="if specified, load prompts from this file",
#)
#parser.add_argument(
#    "--config",
#    type=str,
#    default="configs/stable-diffusion/v1-inference.yaml",
#    help="path to config which constructs model",
#)
#parser.add_argument(
#    "--ckpt",
#    type=str,
#    default="models/ldm/stable-diffusion-v1/model.ckpt",
#    help="path to checkpoint of model",
#)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--random_seed",
    type=bool,
    default=False,
    help="generate a random seed"
)
#parser.add_argument(
#    "--precision",
#    type=str,
#    help="evaluate at this precision",
#    choices=["full", "autocast"],
#    default="autocast"
#)
parser.add_argument(
    "--hardware",
    type=str,
    help="GPU or CPU processing",
    choices=["gpu","cpu"],
    default="gpu"
)
opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path)) + 1

def get_latents_from_seed(samples:int, seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (samples, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

if opt.hardware == "gpu":
    pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider")
elif opt.hardware == "cpu":
    pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx")

if opt.random_seed:
    seed = random.randint(1,1152921504606846975)
    print("seed: "+str(seed))
else:
    seed = opt.seed

"""
prompt: Union[str, List[str]],
height: Optional[int] = 512,
width: Optional[int] = 512,
num_inference_steps: Optional[int] = 50,
guidance_scale: Optional[float] = 7.5, # This is also sometimes called the CFG value
eta: Optional[float] = 0.0,
latents: Optional[np.ndarray] = None,
output_type: Optional[str] = "pil",
"""
batch_size = opt.n_samples
prompt = opt.prompt
# Generate our own latents so that we can provide a seed.
latents = get_latents_from_seed(batch_size ,seed, opt.H, opt.W)
data = [prompt]* batch_size
results = pipe(data, height=opt.H, width=opt.W, num_inference_steps=opt.ddim_steps, guidance_scale=opt.scale, eta=opt.ddim_eta, latents=latents )
#for image in images:
for image in (results.images):
    image.save(os.path.join(sample_path, f"{base_count:05}.png"))
    base_count += 1