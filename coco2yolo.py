import json
import os
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, image_dir, output_label_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)

    os.makedirs(output_label_dir, exist_ok=True)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    img_to_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    for img_id, img in tqdm(images.items()):
        file_name = img["file_name"]
        width, height = img["width"], img["height"]
        anns = img_to_anns.get(img_id, [])

        label_path = os.path.join(output_label_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                bbox = ann["bbox"]  # COCO: [x_min, y_min, width, height]
                x_c = (bbox[0] + bbox[2] / 2) / width
                y_c = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height
                class_id = categories[ann["category_id"]]
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# Run only if this script is run directly (not when imported)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to COCO annotation file")
    parser.add_argument("--imgdir", required=True, help="Path to image folder")
    parser.add_argument("--outdir", required=True, help="Where to save YOLO labels")
    args = parser.parse_args()
    convert_coco_to_yolo(args.json, args.imgdir, args.outdir)