import os
import json
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import classification_report

# ============================
# CONFIGURATION
# ============================
TRAIN_DIR = "/home/smurfy/Desktop/Plant-Doctor/DATASET/tomato/train"
VAL_DIR = "//home/smurfy/Desktop/Plant-Doctor/DATASET/tomato/val"
MODEL_SAVE_PATH = "/home/smurfy/Desktop/Plant-Doctor/MAJOR_PROJECT/TOMATO/MODELS/NEW.h5"
GRAPH_DIR = "./GRAPH"
HISTORY_SAVE_PATH = os.path.join(GRAPH_DIR, "training_history.json")

LABEL_NAMES = [
    "BACTERIAL SPOT", "EARLY BLIGHT", "HEALTHY", "LATE BLIGHT", "LEAF MOLD",
    "SEPTORIA LEAF SPOT", "SPIDER MITE", "TARGET SPOT", "MOSAIC VIRUS", "YELLOW LEAF CURL VIRUS"
]

CATEGORIES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# ============================
# DATA LOADING
# ============================
def load_data(data_dir):
    data = []
    for category in CATEGORIES:
        label = CATEGORIES.index(category)
        path = os.path.join(data_dir, category)
        for img_file in os.listdir(path):
            img = cv.imread(os.path.join(path, img_file), 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (64, 64))
            data.append([img, label])
    return data

train_data = load_data(TRAIN_DIR)
test_data = load_data(VAL_DIR)

random.shuffle(train_data)
random.shuffle(test_data)

X_train = np.array([img for img, _ in train_data]) / 255.0
y_train = np.array([label for _, label in train_data])

X_test = np.array([img for img, _ in test_data]) / 255.0
y_test = np.array([label for _, label in test_data])

one_hot_train = to_categorical(y_train)
one_hot_test = to_categorical(y_test)

Y = [LABEL_NAMES[i] for i in y_train]
Z = [LABEL_NAMES[i] for i in y_test]

# ============================
# PLOTS: CLASS DISTRIBUTION
# ============================
os.makedirs(GRAPH_DIR, exist_ok=True)

plt.figure()
sns.countplot(Y, order=LABEL_NAMES)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Leaf Diseases")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "train_distribution.png"))
plt.close()

plt.figure()
sns.countplot(Z, order=LABEL_NAMES)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Leaf Diseases")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "test_distribution.png"))
plt.close()

# ============================
# MODEL
# ============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================
# TRAINING
# ============================
history = model.fit(X_train, one_hot_train, epochs=75, batch_size=128, validation_split=0.2)

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Save training history
with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(history.history, f, indent=4)
print(f"Training history saved to {HISTORY_SAVE_PATH}")

# ============================
# EVALUATION
# ============================
test_loss, test_acc = model.evaluate(X_test, one_hot_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================
# PLOTS: LOSS AND ACCURACY
# ============================
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "loss_plot.png"))
plt.close()

plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Classifier Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "accuracy_plot.png"))
plt.close()

# ============================
# ROC CURVE
# ============================
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

fpr = {}
tpr = {}
roc_auc = {}
n_classes = 10

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['orange', 'green', 'blue', 'red', 'pink', 'purple', 'brown', 'cyan', 'yellow', 'black']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], label=f'{LABEL_NAMES[i]} AUC = {roc_auc[i]:.3f}')
plt.title('Tomato Leaf Diseases - ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
plt.close()

# ============================
# CONFUSION MATRIX
# ============================
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
plt.close()

# ============================
# CLASSIFICATION REPORT PLOT
# ============================
report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)

# Extract precision, recall, f1-score
metrics_df = pd.DataFrame(report).transpose()
metrics_df = metrics_df.iloc[:-3]  # remove 'accuracy', 'macro avg', 'weighted avg'

plt.figure(figsize=(12, 6))
metrics_df[['precision', 'recall', 'f1-score']].plot(
    kind='bar',
    figsize=(14, 6),
    colormap='Set2'
)
plt.title('Classification Report Metrics per Class')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(ticks=np.arange(len(metrics_df)), labels=metrics_df.index, rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()

classification_report_plot_path = os.path.join(GRAPH_DIR, "classification_report_plot.png")
plt.savefig(classification_report_plot_path)
plt.close()
print(f"Classification report plot saved to {classification_report_plot_path}")

# ============================
# SAVE FINAL ACCURACY TO JSON
# ============================
comparison_dir = os.path.join(GRAPH_DIR, "COMPARISON")
os.makedirs(comparison_dir, exist_ok=True)

# Define accuracy data
comparison_data = {
    "Custom CNN (This Work)": {
        "accuracy": round(test_acc * 100, 2)
    }
}

# Save to JSON
comparison_json_path = os.path.join(comparison_dir, "model_metrics.json")
with open(comparison_json_path, "w") as f:
    json.dump(comparison_data, f, indent=4)

print(f"Final accuracy saved to {comparison_json_path}")