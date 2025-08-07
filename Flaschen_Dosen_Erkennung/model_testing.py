import torch 
from PIL import Image
from matplotlib import pyplot as plt


# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/recycling_detector/weights/best.pt')

# Load images
img1 = Image.open('data/test_bilder/pl_gl_can.jpg')
img2 = Image.open('data/test_bilder/plastic_glass_cans_2.jpg')
img3 = Image.open('data/test_bilder/Plastic-Glass-Cans.jpg')

# Perform inference
results1 = model(img1)
results2 = model(img2)
results3 = model(img3)

# Print results
print(results1.pandas().xyxy[0])  # Results for first image
print(results2.pandas().xyxy[0])  # Results for second image
print(results3.pandas().xyxy[0])  # Results for third image

# Visualize results
results1.show()  # Show results for first image
results2.show()  # Show results for second image
results3.show()  # Show results for third image

# Save results to files
results1.save('results1.jpg')  # Save results for first image
results2.save('results2.jpg')  # Save results for second image
results3.save('results3.jpg')  # Save results for third image