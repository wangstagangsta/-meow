#!/usr/bin/env python3
"""
Script to consolidate all .wav files from the data folder into a single folder.
Recursively searches through all subdirectories and moves all .wav files to a destination folder.
"""

import os
import shutil
from pathlib import Path


def consolidate_wav_files(source_dir="data", dest_dir="data/all_wav_files"):
    """
    Move all .wav files from source_dir (and subdirectories) to dest_dir.
    
    Args:
        source_dir: Root directory to search for .wav files
        dest_dir: Destination directory to move all .wav files to
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files recursively
    wav_files = list(source_path.rglob("*.wav"))
    
    # Filter out files that are already in the destination directory
    wav_files = [f for f in wav_files if dest_path not in f.parents]
    
    if not wav_files:
        print(f"No .wav files found in {source_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files to move")
    print(f"Destination: {dest_path.absolute()}")
    print()
    
    moved_count = 0
    skipped_count = 0
    
    for wav_file in wav_files:
        dest_file = dest_path / wav_file.name
        
        # Handle filename conflicts
        if dest_file.exists():
            # If file already exists, check if it's the same file
            if wav_file.samefile(dest_file):
                print(f"Skipping (already in destination): {wav_file.name}")
                skipped_count += 1
                continue
            else:
                # Different file with same name - append number
                base_name = wav_file.stem
                extension = wav_file.suffix
                counter = 1
                while dest_file.exists():
                    new_name = f"{base_name}_{counter}{extension}"
                    dest_file = dest_path / new_name
                    counter += 1
        
        try:
            # Move the file
            shutil.move(str(wav_file), str(dest_file))
            print(f"Moved: {wav_file.relative_to(source_path)} -> {dest_file.name}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving {wav_file}: {e}")
    
    print()
    print(f"Summary:")
    print(f"  Moved: {moved_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Total: {len(wav_files)} files")


if __name__ == "__main__":
    consolidate_wav_files()

