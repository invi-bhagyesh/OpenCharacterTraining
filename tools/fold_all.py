import os, sys, argparse, subprocess

# allow importing from parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from character.utils import get_model_family

HOME = os.getenv("HOME")
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=str, required=True, choices=["distillation", "introspection"])
parser.add_argument("--models", nargs="+", default=[
    "unsloth/Llama-3.1-8B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
])
args = parser.parse_args()


for model in args.models:
    family = get_model_family(model)
    if args.stage == "distillation":
        command = f"python fold_loras.py --model_name {model} --loras_dir {HOME}/loras/{family}-distillation --save_dir_name distilled"
    else:
        command = f"python fold_loras.py --model_name {model} --model_dir {HOME}/models/distilled --loras_dir {HOME}/loras/{family}-introspection --save_dir_name introspection"
    subprocess.run(command, shell=True)