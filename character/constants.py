import os

HOME = os.getenv("HOME", "")

DATA_PATH = "data"
MODEL_PATH = ""  # empty = use HuggingFace Hub model IDs directly
LORA_PATH = f"{HOME}/loras"
CONSTITUTION_PATH = "constitutions"
