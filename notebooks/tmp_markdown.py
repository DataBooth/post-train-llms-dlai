# %% [markdown]
# # Setup for pre-cache of HF models used in the course lessons

# %% [markdown]
# Requires the HuggingFace CLI tool to be installed e.g. `brew install huggingface-cli`.
# 
# Specify the list of models to be downloaded (if you have already downloaded them they will not be downloaded again.)
# 
# Note: The usual cache location is `~/.cache/huggingface`.

# %%
!ls ~/.cache/huggingface

# %%
lesson_models = [
    "Qwen/Qwen3-0.6B-Base",
    "banghua/Qwen3-0.6B-SFT",
    "HuggingFaceTB/SmolLM2-135Mbanghua/Qwen2.5-0.5B-DPO",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "banghua/Qwen2.5-0.5B-DPO",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "banghua/Qwen2.5-0.5B-GRPO",
]

# %%
# List of models to be downloaded

hf_models_to_download = [
    "smol-ai/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct"
]

# %%
import httpx

def hf_models_to_urls_with_status(model_ids, repo_type="model", timeout=5.0):
    """
    For each model ID, build its Hugging Face Hub URL and check if it exists.

    Args:
        model_ids (list of str): List like ['smol-ai/SmolVLM-256M-Instruct']
        repo_type (str): 'model', 'dataset', or 'space'
        timeout (float): HTTP timeout in seconds

    Returns:
        list of dicts with keys: 'id', 'url', 'exists' (bool), 'status_code' (int)
    """
    base_url = "https://huggingface.co"
    if repo_type not in {"model", "dataset", "space"}:
        raise ValueError(f"Invalid repo_type: {repo_type}")

    type_path = {
        "model": "",          # e.g., /facebook/opt-125m
        "dataset": "/datasets",
        "space": "/spaces"
    }[repo_type]

    results = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for model_id in model_ids:
            url = f"{base_url}{type_path}/{model_id}"
            try:
                response = client.get(url)
                exists = response.status_code == 200
            except httpx.RequestError as e:
                exists = False
                response = None
                print(f"⚠️ Request to {url} failed: {e}")
            results.append({
                "id": model_id,
                "url": url,
                "exists": exists,
                "status_code": response.status_code if response else None
            })
    return results


# %%
hf_models_to_urls_with_status(hf_models_to_download)

# %%
def hf_repo_exists(model_id, repo_type="model"):
    """Check if a Hugging Face model/dataset/space exists."""
    base_url = "https://huggingface.co"
    type_path = {
        "model": "",
        "dataset": "/datasets",
        "space": "/spaces"
    }[repo_type]

    url = f"{base_url}{type_path}/{model_id}"
    try:
        resp = httpx.get(url, follow_redirects=True)
        return resp.status_code == 200
    except httpx.RequestError as e:
        print(f"⚠️ Error checking {url}: {e}")
        return False

# %%
# 🔧 Set dry_run = True to skip downloading
dry_run = False  # toggle this to False when you're ready to download

# Download or preview
n_checked = 0
n_found = 0
n_downloaded = 0

for repo in hf_models_to_download:
    print(f"🔎 Checking: {repo}...", end=" ")
    n_checked += 1
    if hf_repo_exists(repo):
        n_found += 1
        print("✅ Found", end=" ")
        if dry_run:
            print("— dry run (skipped download)")
        else:
            print("— downloading")
            !huggingface-cli download {repo}
            n_downloaded += 1
    else:
        print("❌ Not found — skipped")

print("\n🟩 Summary:")
print(f"  🔍 Checked    : {n_checked}")
print(f"  ✅ Found      : {n_found}")
print(f"  📥 Downloaded : {n_downloaded} {'(dry run)' if dry_run else ''}")

# %%



