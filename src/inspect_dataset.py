from pathlib import Path

DATA_ROOT = Path("data/raw")

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def get_stem_set(folder: Path, allowed_exts=None):
    files = []
    for file in folder.iterdir():
        if file.is_file():
            if allowed_exts is None or file.suffix.lower() in allowed_exts:
                files.append(file)
    return sorted(files), {file.stem for file in files}


def inspect_group(group_path: Path):
    images_path = group_path / "images"
    gt_path = group_path / "ground_truth"

    print(f"\n--- Inspecting group: {group_path.name} ---")

    if not images_path.exists():
        print("ERROR: images folder missing")
        return

    if not gt_path.exists():
        print("ERROR: ground_truth folder missing")
        return

    image_files, image_stems = get_stem_set(images_path, VALID_IMAGE_EXTENSIONS)
    gt_files, gt_stems = get_stem_set(gt_path, {".txt"})

    print(f"Number of image files      : {len(image_files)}")
    print(f"Number of ground truth txt : {len(gt_files)}")

    print("\nImage files:")
    for f in image_files:
        print(f"  {f.name}")

    print("\nGround truth files:")
    for f in gt_files:
        print(f"  {f.name}")

    missing_gt = image_stems - gt_stems
    missing_images = gt_stems - image_stems

    print("\nPairing check:")
    if not missing_gt and not missing_images:
        print("  OK: Every image has a matching ground truth file.")
    else:
        if missing_gt:
            print("  Missing ground truth for image(s):")
            for name in sorted(missing_gt):
                print(f"    {name}")
        if missing_images:
            print("  Missing image for ground truth file(s):")
            for name in sorted(missing_images):
                print(f"    {name}")


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root not found: {DATA_ROOT}")
        return

    group_folders = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])

    print("Dataset groups found:")
    for group in group_folders:
        print(f"  - {group.name}")

    for group in group_folders:
        inspect_group(group)


if __name__ == "__main__":
    main()