import os
import csv
import json
from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ypxyYsKEoGKn71E2jFvq"
)

# Folder with images
image_folder = "data/"
output_folder = "output/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
image_files = [f for f in os.listdir(image_folder) if f.startswith("pic_") and f.endswith((".jpg", ".jpeg", ".png"))]

# Dictionary to store results
all_results = []

# Loop through images
for image_file in sorted(image_files, key=lambda x: int(x.split("_")[1].split(".")[0])):
    image_path = os.path.join(image_folder, image_file)

    # Send request to Roboflow API with image2 as the key and the image file path as the value
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = client.run_workflow(
            workspace_name="spinedetection",
            workflow_id="bottledetectwf",
            images={"image": image_bytes},  # NEW: pass file handle
            #use_cache=True
        )

        print(f"✅ Processed {image_file}")  # Debugging output

        # Store result
        all_results.append({"image": image_file, "predictions": result})

    except Exception as e:
        print(f"❌ Error processing {image_file}: {e}")

# Save results to JSON
with open("results.json", "w") as json_file:
    json.dump(all_results, json_file, indent=4)

# Save results to CSV
with open("results.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "Detection", "Class", "Confidence", "X", "Y", "Width", "Height"])

    for result in all_results:
        image_name = result["image"]
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"❌ Image file {image_name} does not exist.")
            continue

        image = cv2.imread(image_path)

        # Define color mapping for each class
        color_map = {
            "can": (0, 0, 255),       # Red (BGR format)
            "glass": (255, 255, 0),   # Cyan
            "plastic": (0, 255, 0)    # Green
        }

        for detection in result["predictions"][0].get("predictions", {}).get("predictions", []):
            x, y, w, h = int(detection["x"]), int(detection["y"]), int(detection["width"]), int(detection["height"])
            class_name = detection.get("class", "").lower()  # Ensure class name is lowercase
            color = color_map.get(class_name, (255, 255, 255))  # Default to white if class unknown
            cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)

        # Save annotated image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        print("✅ Visualization complete! Processed images saved in 'output/' folder.")

        predictions = result["predictions"][0].get("predictions", {}).get("predictions", [])

        for i, detection in enumerate(predictions):
                writer.writerow([
                    image_name, 
                    i + 1, 
                    detection["class"], 
                    detection["confidence"], 
                    detection["x"], 
                    detection["y"], 
                    detection["width"], 
                    detection["height"],
                ])

print("🎉 Inference complete! Results saved to results.json and results.csv")
