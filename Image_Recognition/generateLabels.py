import pandas as pd

# Load your results file
df = pd.read_csv("Image_Recognition/image_recognition_results.csv")

def assign_label(row):
    tags = str(row.get("gemini_tags", "")).lower()

    if "applique" in tags:
        return "applique"
    if "pantograph" in tags or "edge to edge" in tags or "e2e" in tags:
        return "pantograph"
    if "quilt" in tags or "quilting" in tags:
        return "quilting"
    return "embroidery"

df["assigned_label"] = df.apply(assign_label, axis=1)

# Save file
df[["image_file", "assigned_label"]].to_csv(
    "Image_Recognition/labeled_images_for_clarifai.csv",
    index=False
)

print("Saved labeled_images_for_clarifai.csv")
