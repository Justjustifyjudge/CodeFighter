from modelscope import snapshot_download

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
local_dir = "./Qwen2.5-Coder-32B-Instruct"

snapshot_download(repo_id=model_name, local_dir=local_dir)