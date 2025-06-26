import os
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras.applications import (
    MobileNetV2,
    DenseNet121,
    ResNet50
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import joblib

# ====================
# CONFIGURATION
# ====================
class Config:
    DATASET_DIR = "/home/smurfy/Desktop/Plant-Doctor/DATASET/Bean_Dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    VAL_SPLIT = 0.2
    SEED = 42
    PLOT_DIR = "./GRAPH"
    HISTORY_DIR = "./HISTORY"

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
        return features, np.argmax(labels, axis=1)

# ====================
# MODEL TRAINER
# ====================
class MLModelTrainer:
    def __init__(self, model_name):
        if model_name == "svm":
            self.model = SVC(kernel='rbf', C=1.0, probability=True)
        elif model_name == "rf":
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Unsupported model")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("\nAccuracy:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        return acc, y_pred

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")

# ====================
# PLOTTING & LOGGING
# ====================
def save_confusion_matrix(y_true, y_pred, class_names, title, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{title}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")

# ====================
# MAIN EXECUTION
# ====================
def main():
    config = Config()
    dataset, class_names = load_data(config)
    base_models = {"MobileNetV2": MobileNetV2, "DenseNet121": DenseNet121, "ResNet50": ResNet50}

    detailed_results = []
    bean_results = {}  #Dictionary to store all accuracies for BEAN

    for model_name, base_model_fn in base_models.items():
        print(f"\n--- Using {model_name} for feature extraction ---")
        extractor = FeatureExtractor(base_model_fn, config.IMG_SIZE)
        X, y = extractor.extract(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)

        for ml_model in ["svm", "rf"]:
            tag = f"{model_name}_{ml_model.upper()}"
            print(f"\nTraining {ml_model.upper()} classifier with {model_name} features")

            trainer = MLModelTrainer(ml_model)
            trainer.train(X_train, y_train)
            acc, y_pred = trainer.evaluate(X_test, y_test)

            # Save accuracy to bean_results
            bean_results[f"BEAN_{tag}_ACCURACY"] = round(acc, 4)

            # Save confusion matrix
            plot_dir = os.path.join(config.PLOT_DIR, model_name, ml_model.upper())
            save_confusion_matrix(y_test, y_pred, class_names, tag, plot_dir)

            # Save model
            model_path = os.path.join("MODELS", model_name, ml_model.upper(), f"{tag}.joblib")
            trainer.save(model_path)

            # Optional: save per-model training history
            history_path = os.path.join(config.HISTORY_DIR, f"{tag}_history.json")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump({"final_accuracy": acc}, f, indent=4)

            detailed_results.append({
                "Model": model_name,
                "Classifier": ml_model.upper(),
                "Accuracy": round(acc, 4),
                "Model_Path": model_path,
                "Confusion_Matrix": os.path.join(plot_dir, f"{tag}_confusion_matrix.png"),
                "History_File": history_path
            })

    # Save bean_results as single JSON for the plant
    os.makedirs("BEAN", exist_ok=True)
    with open("BEAN/BEAN_accuracy.json", "w") as f:
        json.dump(bean_results, f, indent=4)
    print("Saved consolidated BEAN_accuracy.json file.")

    # Save all logs (unchanged)
    summary_dir = os.path.join(config.PLOT_DIR, "Comparison")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "model_metrics.json"), "w") as f:
        json.dump(detailed_results, f, indent=4)
    pd.DataFrame(detailed_results).to_csv(os.path.join(summary_dir, "model_metrics.csv"), index=False)
    print("Saved final JSON and CSV summaries.")
if __name__ == "__main__":
    main()
