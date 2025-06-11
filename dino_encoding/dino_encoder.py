import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

def get_embedding(image_folder):
    img_files = sorted(os.listdir(image_folder))
    embeddings = []

    for img_file in img_files:
        image_path = os.path.join(image_folder, img_file)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs)
            cls_token = output.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_token.squeeze(0))

    if len(embeddings) == 0:
        return None
    embeddings = torch.stack(embeddings)
    object_embedding = embeddings.mean(dim=0)  # Final embedding vector [768]
    return object_embedding

dataset_root = "/iris/u/duyi/cs231a/get_a_grip/data/dataset/small_random/nerfdata"   # Change dataset_root to be the direct file path to "nerf_data"
object_embedding_dict = {}

for folder in sorted(os.listdir(dataset_root)):
    if folder.startswith("."):
        continue

    folder_path = os.path.join(dataset_root, folder)
    if not os.path.isdir(folder_path):
        continue  

    image_folder = os.path.join(folder_path, "images")
    if not os.path.isdir(image_folder):
        print(f"Skipping {folder}: 'images' folder not found.")
        continue

    print(f"Processing: {folder}")
    embedding = get_embedding(image_folder)
    if embedding is not None:
        object_embedding_dict[folder] = embedding.cpu()
    else:
        print(f"Warning: No valid images found in {image_folder}")

torch.save(object_embedding_dict, "dino_object_embeddings.pt")
print("Saved embeddings to dino_object_embeddings.pt")

