import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import MobileNetV2, DenseNet121, ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models, layers, optimizers, callbacks
import h5py
from sklearn.metrics import classification_report

# ====================
# CONFIGURATION
# ====================
class Config:
    DATASET_DIR = "/home/smurfy/Desktop/Plant-Doctor/DATASET/Potato"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    VAL_SPLIT = 0.2
    SEED = 42
    PLOT_DIR = "./GRAPH"
    MODEL_DIR = "./MODELS"

# ====================
# GPU MEMORY LIMITATION HANDLING
# ====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ====================
# DATA PIPELINE
# ====================
def load_data(config):
    train_ds = image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=config.VAL_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    val_ds = image_dataset_from_directory(
        config.DATASET_DIR,
        validation_split=config.VAL_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    return train_ds.concatenate(val_ds), class_names

# ====================
# FEATURE EXTRACTOR
# ====================
class FeatureExtractor:
    def __init__(self, base_model_fn, input_shape):
        self.model = base_model_fn(
            input_shape=input_shape + (3,),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.model.trainable = False

    def extract(self, dataset):
        features, labels = [], []
        dataset = dataset.unbatch().batch(4)

        for batch_images, batch_labels in tqdm(dataset, desc="Extracting Features", colour="white", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch_images)
            batch_features = self.model(preprocessed, training=False).numpy()
            features.append(batch_features)
            labels.append(batch_labels.numpy())

        features = np.concatenate(features)
        labels = np.concatenate(labels)
        return features, labels

# ====================
# TRAINER
# ====================
class MLModelTrainer:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10):
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True)
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return acc, report
    def save(self, save_path, model_name):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

        with h5py.File(save_path, 'a') as f:
            hist_group = f.create_group(f"training_history/{model_name}")
            for key, val in self.history.history.items():
                hist_group.create_dataset(key, data=val)
            hist_group.attrs['epochs'] = len(self.history.history['loss'])
        print(f"Training history saved under group 'training_history/{model_name}' in {save_path}")

# ====================
# PLOTTING
# ====================
def save_accuracy_plot(model_accuracies, title, save_path):
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.barh(models, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy plot saved to {save_path}")
import json
import seaborn as sns

def plot_classification_heatmap(report, labels, title, save_path):
    report_matrix = np.array([[
        report[str(i)]['precision'],
        report[str(i)]['recall'],
        report[str(i)]['f1-score']
    ] for i in range(len(labels))])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_matrix, annot=True, fmt=".2f", xticklabels=['Precision', 'Recall', 'F1-Score'], yticklabels=labels, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Classification report heatmap saved to {save_path}")

def main():
    config = Config()
    dataset, class_names = load_data(config)
    num_classes = len(class_names)

    base_models = {
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "ResNet50": ResNet50
    }

    all_model_accuracies = {}

    for model_name, base_model_fn in base_models.items():
        print(f"\n--- Using {model_name} for feature extraction ---")
        extractor = FeatureExtractor(base_model_fn, config.IMG_SIZE)
        X, y = extractor.extract(dataset)
        y = tf.keras.utils.to_categorical(np.argmax(y, axis=1), num_classes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=config.SEED)

        trainer = MLModelTrainer(input_shape=X.shape[1:], num_classes=num_classes)
        trainer.train(X_train, y_train, X_val, y_val, epochs=10)
        acc, report = trainer.evaluate(X_test, y_test)

        # Save model
        model_save_path = os.path.join(config.MODEL_DIR, model_name, f"{model_name}_KERAS.h5")
        trainer.save(model_save_path, model_name)

        # Store accuracy only in the JSON
        all_model_accuracies[f"{model_name}_KERAS"] = round(acc, 4)

        # Save classification report heatmap
        report_heatmap_path = os.path.join(config.PLOT_DIR, "Comparison", f"{model_name}_report_heatmap.png")
        os.makedirs(os.path.dirname(report_heatmap_path), exist_ok=True)
        plot_classification_heatmap(report, class_names, f"{model_name} Classification Report", report_heatmap_path)

    # === Save accuracy bar plot ===
    comparison_dir = os.path.join(config.PLOT_DIR, "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_plot_path = os.path.join(comparison_dir, "all_model_accuracies.png")
    save_accuracy_plot(all_model_accuracies, "All Model Accuracies", comparison_plot_path)

    # === Save model accuracies to JSON ===
    json_path = os.path.join(comparison_dir, "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump(
            {k: {"accuracy": float(v)} for k, v in all_model_accuracies.items()},
            f, indent=4
        )
    print(f"Model accuracies saved to {json_path}")

if __name__ == "__main__":
    main()