import concurrent.futures
import os

import requests
from tqdm import tqdm

persistent_id = "doi:10.7910/DVN/N4LNZD"
version = "2.0"
dataset_api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/versions/{version}?persistentId={persistent_id}"
download_base_url = "https://dataverse.harvard.edu/api/access/datafile"

download_dir = "/Volumes/MARCREYES/violence_detection/rar"
os.makedirs(download_dir, exist_ok=True)

print("Fetching dataset metadata...")
response = requests.get(dataset_api_url)
response.raise_for_status()
data = response.json()

files = data["data"]["files"]
print(f"Found {len(files)} files. Starting download...\n")


def download_file(file):
    filename = file["dataFile"]["filename"]
    file_id = file["dataFile"]["id"]
    file_path = os.path.join(download_dir, filename)

    print(f"Downloading {filename}...")
    with requests.get(f"{download_base_url}/{file_id}", stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

    print(f"Saved to {file_path}\n")
    return filename


max_workers = min(5, len(files))
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_file = {executor.submit(download_file, file): file for file in files}

    for future in concurrent.futures.as_completed(future_to_file):
        try:
            filename = future.result()
        except Exception as exc:
            print(f"Download failed: {exc}")

print("✅ All files downloaded successfully.")
