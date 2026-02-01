import os, torch, clip, shutil
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cpu")

domains = [
    "people_daily_life","transport_and_vehicles","animals_wildlife","sports_recreation",
    "food_drink","home_furniture","electronics_devices","appliances_kitchen",
    "outdoor_nature_landscapes","urban_street_infrastructure","fashion_accessories",
    "plants_gardens","toys_kids","tools_office_supplies","art_decor_objects"
]

with torch.no_grad():
    text_feat = model.encode_text(clip.tokenize(domains))
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

img_dir = r"D:\Dataset\val2017\val2017"
output_dir = "D:\Dataset\classified_images"

os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(img_dir):
    try:
        img_path = os.path.join(img_dir, f)
        img_pil = Image.open(img_path).convert("RGB")

        img = preprocess(img_pil).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            best_idx = (feat @ text_feat.T).softmax(-1).argmax().item()
            pred_domain = domains[best_idx]

        domain_folder = os.path.join(output_dir, pred_domain)
        os.makedirs(domain_folder, exist_ok=True)

        shutil.copy(img_path, os.path.join(domain_folder, f))

        print(f"Saved {f} → {pred_domain}")
    except Exception as e:
        print("Error:", e)
        continue
