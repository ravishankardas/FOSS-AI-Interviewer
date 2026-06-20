import platform
import psutil
import math
from loguru import logger
import os
from huggingface_hub import hf_hub_download
import yaml

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 

def pick_model(system_ram_in_gb):
    """
    pick the model based on system RAM
    """

    logger.info(f"System RAM is: {system_ram_in_gb}")
    model_name_to_download = ""
    if system_ram_in_gb < 8:
        model_name_to_download = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    elif 8 <= system_ram_in_gb <= 16:
        model_name_to_download = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    elif system_ram_in_gb > 16:
        model_name_to_download = "Llama-3.2-7B-Instruct-Q8_0.gguf"

    index = model_name_to_download.index("Instruct")

    repo_id = f"bartowski/{model_name_to_download[:index + 9]}GGUF"

    return (repo_id, model_name_to_download)

def download_model(repo_id, filename):
    """
    download the correct model if not present
    """
    model_path = f"models/{filename}"
    if os.path.exists(model_path):
        logger.info(f"model already exists: {model_path}")
        return model_path
    
    logger.info(f"downloading: {filename}")
    
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="models/"
    )

    logger.info(f"model: {filename}, downloaded in: {local_path}")

    return local_path
    
def update_config(model_path):
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["llm"]["model_path"] = model_path

    logger.info("updating the local_model path in config")

    with open(config_path, "w") as f:
        yaml.dump(config, f)



if __name__ == "__main__":
    system_ram_in_gb = math.ceil(psutil.virtual_memory().total / (1024 ** 3))
    repo_id, file_name = pick_model(system_ram_in_gb)
    local_path = download_model(repo_id, file_name)

    update_config(local_path)

