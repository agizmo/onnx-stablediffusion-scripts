from diffusers import OnnxStableDiffusionImg2ImgPipeline
import numpy as np
import argparse, os, random, csv,warnings
import PIL
from PIL import Image

def get_latents_from_seed(samples:int, seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (samples, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

def load_img(path, width:int, height:int):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    if w > width or h > height:
        image.thumbnail((width,height), resample=PIL.Image.Resampling.BICUBIC)
        w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    return image
    #image = np.array(image).astype(np.float32) / 255.0
    #image = image[None].transpose(0, 3, 1, 2)
    #image = torch.from_numpy(image)
    #return 2.*image - 1.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="The prompt to render"
)
parser.add_argument(
      "--init_img",
      type=str,
      nargs="?",
      help="path to the input image"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="Default: outputs/img2img-samples. Directory to write results to",
    default="outputs/img2img-samples"
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
    default=1,
    help="Default: 1. How many samples to produce for each given prompt. A.k.a. batch size",
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
    "--model",
    type=str,
    default="./stable_diffusion_onnx",
    help="Default: ./stable_diffusion_onnx. Path to the model folder",
)
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
    "--negative_prompt",
    type=str,
    default="",
    help="Default: ''. Prompts to not guide the image generation"
)
parser.add_argument(
    "--hardware",
    type=str,
    help="Default: directml. DirectML or CPU processing",
    choices=["directml","cpu"],
    default="directml"
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
    help="Default: False. Create a log file detailing the image parameters. CSV file is saved in outdir",
    default=False
)

opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

#assert os.path.isfile(opt.init_img)
#init_image = Image.open(opt.init_img)
init_image = load_img(opt.init_img, opt.W, opt.H)

logfilename = "log.csv"
logpath = os.path.join(outpath,logfilename)
if opt.log and not os.path.exists(logpath):
    fields = ['image','ddim_steps','ddim_eta','H','W','n_samples','scale','seed','prompt','negative_prompt','init_img']
    with open(logpath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path)) + 1

if opt.hardware == "directml":
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.model, provider="DmlExecutionProvider")
elif opt.hardware == "cpu":
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.model, provider="CPUExecutionProvider")

for lp in range(opt.loop):
    if opt.random_seed:
        seed = random.randint(1,(2**32))
        print("seed: "+str(seed))
    else:
        seed = opt.seed

    batch_size = opt.n_samples
    prompt = opt.prompt
    nprompt = opt.negative_prompt
    # Generate our own latents so that we can provide a seed.
    latents = get_latents_from_seed(batch_size ,seed, opt.H, opt.W)
    data = [prompt]* batch_size
    ndata = [nprompt]* batch_size
    steps = int(opt.ddim_steps / 0.8) #Don't know why but the img2img pipeline only runs 8/10 of the steps supplied. 
    results = pipe(data, init_image=init_image, height=opt.H, width=opt.W, num_inference_steps=steps, guidance_scale=opt.scale, eta=opt.ddim_eta, latents=latents, negative_prompt=ndata )
    #for image in images:
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            print(f"{base_count:05}.png image flagged as having nsfw content. Try a different seed or ddim_step count.")
        
        results.images[i].save(os.path.join(sample_path, f"{base_count:05}.png"))

        row = [f"{base_count:05}.png",opt.ddim_steps,opt.ddim_eta,opt.H,opt.W,opt.n_samples,opt.scale,seed,opt.prompt,opt.negative_prompt,opt.init_img]
        if opt.log:
            with open(logpath, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)

        base_count += 1
