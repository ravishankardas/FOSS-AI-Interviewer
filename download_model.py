import math
import os
import yaml # type: ignore
import psutil # type: ignore
from loguru import logger # type: ignore
from huggingface_hub import hf_hub_download # type: ignore

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def pick_model(system_ram_in_gb):
    logger.info(f"System RAM is: {system_ram_in_gb}")
    if system_ram_in_gb < 8:
        model_name = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    elif 8 <= system_ram_in_gb <= 16:
        model_name = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    else:
        model_name = "Llama-3.2-7B-Instruct-Q8_0.gguf"
    index = model_name.index("Instruct")
    repo_id = f"bartowski/{model_name[:index + 9]}GGUF"
    return repo_id, model_name

def download_model(repo_id, filename):
    model_path = f"models/{filename}"
    if os.path.exists(model_path):
        logger.info(f"model already exists: {model_path}")
        return model_path
    logger.info(f"downloading: {filename}")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models/")
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



def update_stt_config(model_name):
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["stt"]["model_path"] = model_name
    logger.info("updating the stt_model path in config")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
def get_whisper_model_name(system_ram):
    model_name = ""
    if system_ram < 4:
        model_name = "tiny"
    elif 4 <= system_ram <= 8:
        model_name = "small"
    elif 8 < system_ram <= 16:
        model_name = "medium"
    else:
        model_name = "large-v3"

    return model_name


def download_tts_model():

    local_dir = "models/"

    file_name = f"{local_dir}en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    file_name_json = f"{file_name}.json"
    repo_id = "rhasspy/piper-voices"

    if os.path.exists(file_name) and os.path.exists(file_name_json):
        logger.info("TTS model already exists, skipping...")
        return
    
    hf_hub_download(
        repo_id = repo_id,
        filename = file_name,
        local_dir = local_dir
    )

    hf_hub_download(
        repo_id=repo_id,
        filename=file_name_json,
        local_dir=local_dir
    )

    logger.info("TTS model downloaded successfully")



                    
if __name__ == "__main__":
    system_ram_in_gb = math.ceil(psutil.virtual_memory().total / (1024 ** 3))
    # repo_id, file_name = pick_model(system_ram_in_gb)
    # local_path = download_model(repo_id, file_name)
    # update_config(local_path)


    # whisper_model = get_whisper_model_name(system_ram_in_gb)
    # logger.info(f"whisper_model: {whisper_model}")
    # update_stt_config(whisper_model)

    download_tts_model()
