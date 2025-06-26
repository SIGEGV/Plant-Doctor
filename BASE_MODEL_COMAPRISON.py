import os
import json
import matplotlib.pyplot as plt

# Step 1: Base paper models and their accuracies
# base_models = [
#     "EfficientB5Net", "InceptionV3Net", "DenseNet201", "AlexNet", "ResNet101", "VGG19",
#     "GOGLENet", "PP-DCLNet", "Dilated CNN and Elman", "ResNet50 + EC-BAM", "FasterRCNN",
#     "DCNN (RESNET)", "DCNN + YOLO", "YOLO-CNN", "D.T + CNN", "Customized GA", "BPN-FA",
#     "U-Net", "Modified U-Net", "K.S", "RS-Net", "ResNeXt", "MobileNet", "DenseNet121",
#     "PseuCNN", "Omni-FBPDCNN", "TL FBP-CNN", "A multiinput ResNet", "GLCM + Haralick",
#     "KMC + GLCM + CNN", "TL (InceptionV3, MobileNetV2, DenseNet121)", 
#     "TL (ResNet101, VGG16, VGG19, GoogleNet, AlexNet, ResNet50)",
#     "SVM", "CNN", "CNN", "SVM", "KNN", "KNN", "RF", "RF"
# ]

# base_accuracies = [
#     94.51, 90.53, 89.55, 92.70, 96.00, 96.10,
#     95.71, 96.00, 95.71, 99.97, 94.50, 98.12,
#     99.89, 99.90, 99.92, 94.86, 95.63, 98.35,
#     98.49, 97.43, 91.61, 77.56, 76.16, 74.53,
#     79.32, 94.29, 94.60, 97.48, 98.30, 97.63,
#     98.39, 98.90, 99.92, 100.00, 95.29, 99.00,
#     96.00, 94.50, 91.10, 96.00
# ]

# Step 2: Your model accuracy collection
your_models = []
your_accuracies = []
your_colors = []

# Define paths to model_metrics.json for each plant
model_metric_paths = {
    'BEAN': '/home/smurfy/Desktop/Plant-Doctor/Model_Training/BEAN/GRAPH/Comparison/model_metrics.json',
    'GRAPE': '/home/smurfy/Desktop/Plant-Doctor/Model_Training/GRAPE/GRAPH/COMPARISION/GRAPE_accuracy.json',
    'POTATO': '/home/smurfy/Desktop/Plant-Doctor/Model_Training/POTATO/GRAPH/Comparison/model_metrics.json',
    'TOMATO': '/home/smurfy/Desktop/Plant-Doctor/Model_Training/TOMATO/GRAPH/COMPARISON/model_metrics.json'
}

# Step 3: Load and parse metrics
for plant, path in model_metric_paths.items():
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        continue

    with open(path, 'r') as f:
        metrics = json.load(f)

    if plant == 'BEAN':
        # BEAN is a list of dicts
        for item in metrics:
            model_name = f"{item['Model']} + {item['Classifier']} ({plant})"
            acc = round(item.get('Accuracy', 0) * 100, 2)
            your_models.append(model_name)
            your_accuracies.append(acc)
            your_colors.append('red')
    else:
        # Others are dicts with 'accuracy' keys
        for model_key, stats in metrics.items():
            model_name = f"{model_key} ({plant})"
            acc = round(stats.get('accuracy', 0) * 100, 2)
            your_models.append(model_name)
            your_accuracies.append(acc)
            your_colors.append('red')

# Step 4: Combine all data
all_models =  your_models
all_accuracies = your_accuracies
all_colors =    your_colors

# Step 5: Plot and save
plt.figure(figsize=(28, 10))
bars = plt.bar(range(len(all_models)), all_accuracies, color=all_colors)
plt.xticks(range(len(all_models)), all_models, rotation=75, ha='right', fontsize=7)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of  Models")

# Annotate each bar
for bar, acc in zip(bars, all_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{acc:.2f}%', 
             ha='center', fontsize=6, rotation=90)

# Save figure
plt.tight_layout()
plt.savefig("MODELS_ACCURACY_ANALYSIS.png", dpi=300, bbox_inches='tight')
print("Saved: BASE_MODEL_ACCURACY_COMPARISON.png")
