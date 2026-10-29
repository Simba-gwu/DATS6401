"""Create a small, reproducible EuroSAT sample for the Week 8 Streamlit app.

Run from the week8 directory: python prepare_week8_sample.py
Requires the original dataset at data/EuroSAT_RGB/<category>/*.jpg.
"""

from pathlib import Path
import csv
import random
import shutil

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "data" / "EuroSAT_RGB"
SAMPLE = BASE / "data" / "sample"
COUNTS_CSV = BASE / "data" / "class_counts.csv"
IMAGES_PER_CLASS = 30
SEED = 42


def main():
    if not SOURCE.is_dir():
        raise SystemExit(
            f"Dataset not found at {SOURCE}\n"
            "Check that the 10 category folders are inside week8/data/EuroSAT_RGB."
        )

    categories = sorted(folder for folder in SOURCE.iterdir() if folder.is_dir())
    if not categories:
        raise SystemExit(f"No category folders found in {SOURCE}")

    # Only replace the generated sample. Never touch the original EuroSAT_RGB folder.
    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)
    SAMPLE.mkdir(parents=True)
    COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    records = []

    for folder in categories:
        paths = sorted(folder.glob("*.jpg"))
        if not paths:
            print(f"Skipping {folder.name}: no JPG images found")
            continue

        selected = sorted(rng.sample(paths, min(IMAGES_PER_CLASS, len(paths))))
        destination = SAMPLE / folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in selected:
            # EuroSAT RGB images are already just 64 x 64 pixels.
            # Copy a small representative sample without resizing or altering colors.
            shutil.copy2(image_path, destination / image_path.name)

        records.append({
            "category": folder.name,
            "total_images": len(paths),
            "sampled_images": len(selected),
        })
        print(f"{folder.name}: {len(paths)} original images -> {len(selected)} sampled")

    if not records:
        raise SystemExit("No images found; sample not created.")

    with COUNTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "total_images", "sampled_images"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nOriginal images: {sum(r['total_images'] for r in records):,}")
    print(f"Sample images:   {sum(r['sampled_images'] for r in records):,}")
    print(f"Created {COUNTS_CSV.relative_to(BASE)} and data/sample/")
    print("Keep data/EuroSAT_RGB on your Mac; upload only data/sample to GitHub.")


if __name__ == "__main__":
    main()
