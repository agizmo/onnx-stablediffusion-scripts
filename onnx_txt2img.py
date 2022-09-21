from diffusers import StableDiffusionOnnxPipeline
import numpy as np
import argparse, os, random, csv,warnings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="The prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="Default: outputs/txt2img-samples. Directory to write results to",
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
    help="Default: 50. Number of ddim sampling steps",
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
    help="Default: 0.0. ddim eta (eta=0.0 corresponds to deterministic sampling)",
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
    help="Default: 512. Image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="Default: 512. Image width, in pixel space",
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
    default=2,
    help="Default: 2. How many samples to produce for each given prompt. A.k.a. batch size",
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
    help="Default: 7.5. Unconditional guidance scale",
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
    help="Default: 42. The seed (for reproducible sampling)",
)
parser.add_argument(
    "--random_seed",
    type=bool,
    default=False,
    help="Default: False. Generate a random seed value"
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
    help="Default: gpu. GPU or CPU processing",
    choices=["gpu","cpu"],
    default="gpu"
)
parser.add_argument(
    "--loop",
    type=int,
    help="Default: 1. How many times to loop through. USE WITH --random_seed flag",
    default=1
)
parser.add_argument(
    "--log",
    type=bool,
    help="Default: False. Create a logfile detailing the image parameters",
    default=False
)

opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

logfilename = "log.csv"
logpath = os.path.join(outpath,logfilename)
if opt.log and not os.path.exists(logpath):
    fields = ['image','ddim_steps','ddim_eta','H','W','n_samples','scale','seed','prompt']
    with open(logpath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

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
    pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx", provider="CPUExecutionProvider")

if opt.random_seed:
    seed = random.randint(1,(2**32))
    print("seed: "+str(seed))
else:
    seed = opt.seed

batch_size = opt.n_samples
prompt = opt.prompt
for lp in range(opt.loop):
    # Generate our own latents so that we can provide a seed.
    latents = get_latents_from_seed(batch_size ,seed, opt.H, opt.W)
    data = [prompt]* batch_size
    results = pipe(data, height=opt.H, width=opt.W, num_inference_steps=opt.ddim_steps, guidance_scale=opt.scale, eta=opt.ddim_eta, latents=latents )
    #for image in images:
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            print(f"{base_count:05}.png image flagged as having nsfw content. Try a different seed or ddim_step count.")
        
        results.images[i].save(os.path.join(sample_path, f"{base_count:05}.png"))

        row = [f"{base_count:05}.png",opt.ddim_steps,opt.ddim_eta,opt.H,opt.W,opt.n_samples,opt.scale,opt.seed,opt.prompt]
        if opt.log:
            with open(logpath, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)

        base_count += 1