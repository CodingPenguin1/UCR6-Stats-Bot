from zipfile import ZipFile
from pathlib import Path
import subprocess
import pandas as pd
from rich.progress import track
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data")
DECODER_PATH = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/src/decoder/r6-dissect/r6-dissect")


def _extract_match_zip(match_zip: Path, data_dir: Path = DATA_DIR):
    with ZipFile(match_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir / "1_replay_folders")


def _decode_match_folder(match_dir: Path, decoder_path: Path = DECODER_PATH):
    # If already decoded, skip
    processed_files = [f"{file.stem}.json" for file in (DATA_DIR / "2_decoded_replays").iterdir()]
    if f"{match_dir.stem}.json" in processed_files:
        return

    json_path = f"{(DATA_DIR / '2_decoded_replays') / match_dir.stem}.json"
    command = f"{decoder_path} {match_dir} -o {json_path}"
    subprocess.run(command, shell=True, capture_output=True, text=True)
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def decode_matches(data_dir: Path = DATA_DIR, decoder_path: Path = DECODER_PATH):
    # Step 1: Extract the match zips
    zip_files = list((data_dir / "0_replay_zips").iterdir())
    for match_zip in track(zip_files, description="Extracting match zips", total=len(zip_files)):
        _extract_match_zip(match_zip)

    # Step 2: Decode the match folders
    match_folders = list((data_dir / "1_replay_folders").iterdir())
    futures = []
    with ThreadPoolExecutor(8) as executor:
        for match_folder in match_folders:
            futures.append(executor.submit(_decode_match_folder, match_folder, decoder_path))
        for future in track(as_completed(futures), total=len(futures), description="Decoding match folders"):
            future.result()

    # for match_folder in track(match_folders, description="Decoding match folders", total=len(match_folders)):
    #     if f"{match_folder.stem}.xlsx" not in (data_dir / "2_decoded_replays").iterdir():
    #         _decode_match_folder(match_folder, decoder_path)

    # Step 3: Make the tables we want


if __name__ == "__main__":
    decode_matches()
