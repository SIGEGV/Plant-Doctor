import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import MobileNetV2, DenseNet121, ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models, layers, callbacks
import seaborn as sns
import json

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

        for batch_images, batch_labels in tqdm(dataset, desc="Extracting Features"):
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch_images)
            batch_features = self.model(preprocessed, training=False).numpy()
            features.append(batch_features)
            labels.append(batch_labels.numpy())

        features = np.concatenate(features)
        labels = np.argmax(np.concatenate(labels), axis=1)
        return features, labels

# ====================
# CLASSIFIER WRAPPER
# ====================
class MLModelTrainer:
    def __init__(self, model_type, input_shape=None, num_classes=None):
        self.model_type = model_type
        if model_type == "svm":
            self.model = SVC(probability=True)
        elif model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_type == "cnn":
            if input_shape is None or num_classes is None:
                raise ValueError("CNN requires input_shape and num_classes")
            self.model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_classes, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            raise ValueError("Unsupported model type")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model_type == "cnn":
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                callbacks=[early_stop],
                verbose=1
            )
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.model_type == "cnn":
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
        else:
            y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, report, y_pred

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.model_type == "cnn":
            self.model.save(save_path)
        else:
            joblib.dump(self.model, save_path)

# ====================
# HEATMAP PLOTTING
# ====================
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

# ====================
# MAIN EXECUTION
# ====================
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
        print(f"\n--- Using {model_name} ---")
        extractor = FeatureExtractor(base_model_fn, config.IMG_SIZE)
        X, y = extractor.extract(dataset)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=config.SEED)

        for clf in ["svm", "rf", "cnn"]:
            print(f"Training {clf.upper()}...")
            if clf == "cnn":
                trainer = MLModelTrainer(clf, input_shape=X.shape[1:], num_classes=num_classes)
                trainer.train(X_train, y_train, X_val, y_val)
            else:
                trainer = MLModelTrainer(clf)
                trainer.train(X_train, y_train)

            acc, report, y_pred = trainer.evaluate(X_test, y_test)
            print(f"{clf.upper()} Accuracy: {acc:.4f}")

            sub_dir = os.path.join(config.PLOT_DIR, model_name, clf.upper())
            os.makedirs(sub_dir, exist_ok=True)
            plot_path = os.path.join(sub_dir, f"{model_name}_{clf}_report.png")
            plot_classification_heatmap(report, class_names, f"{model_name} - {clf.upper()} Report", plot_path)

            if clf == "cnn":
                save_path = os.path.join(config.MODEL_DIR, model_name, f"CNN_model.h5")
            else:
                save_path = os.path.join(config.MODEL_DIR, model_name, f"{clf.upper()}_model.pkl")

            trainer.save(save_path)
            all_model_accuracies[f"{model_name}_{clf.upper()}"] = acc

    comparison_dir = os.path.join(config.PLOT_DIR, "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(list(all_model_accuracies.keys()), list(all_model_accuracies.values()), color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('All Model Accuracies')
    plt.grid(True, axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "all_model_accuracies.png"))
    plt.close()

    json_path = os.path.join(comparison_dir, "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump(
            {k: {"accuracy": float(v)} for k, v in all_model_accuracies.items()},
            f, indent=4
        )
    print(f"Model accuracies saved to {json_path}")

if __name__ == "__main__":
    main()
