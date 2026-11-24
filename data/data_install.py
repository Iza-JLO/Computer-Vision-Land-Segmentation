from pathlib import Path
from datasets import load_dataset
import csv

DATASET_ID = "nikolkoo/SatelliteSegmentation"
ROOT = Path.home() / "data" / "SatelliteSegmentation"
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# Descargar dataset

print("Conectando con Hugging Face")
ds = load_dataset(DATASET_ID, split="train")
print(f"\nImágenes encontradas: {len(ds)}")
print(f"Columnas: {ds.column_names}")


# Descargar imágenes y máscaras
manifest = []
for i, sample in enumerate(ds):
    image = sample["image"]
    mask = sample["mask"]

    image_name = f"{i:04d}.png"
    mask_name = f"{i:04d}_mask.png"

    image_path = IMAGE_DIR / image_name
    mask_path = MASK_DIR / mask_name
    image.convert("RGB").save(image_path)

    # Conservar los valores originales de las etiquetas
    mask.save(mask_path)

    manifest.append({
        "id": i,
        "image": image_name,
        "mask": mask_name,
        "width": image.width,
        "height": image.height,
        "mask_mode": mask.mode
    })
    if (i + 1) % 100 == 0:
        print(f"Descargadas: {i + 1}/{len(ds)}")


# Guardar manifiesto
with open(ROOT / "manifest.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=manifest[0].keys()
    )
    writer.writeheader()
    writer.writerows(manifest)


print("\nDescarga completada.")
print(f"Imágenes: {len(list(IMAGE_DIR.glob('*.png')))}")
print(f"Máscaras: {len(list(MASK_DIR.glob('*.png')))}")
print(f"Directorio: {ROOT}")