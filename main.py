import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import argparse
from features import extract_tda_features
from forest import get_random_forest, train_random_forest_with_history, evaluate_random_forest, save_random_forest, plot_training_history
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# -------- Argument parsing --------
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, required=True, help='Directory where data is stored')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run the model (recommended: 10-15)')
parser.add_argument('--trees_per_epoch', type=int, default=10, help='Number of trees to add per epoch (default: 10)')
args = parser.parse_args()

data_dir = args.data_directory
num_epochs = args.epochs
trees_per_epoch = args.trees_per_epoch

# Calculate total trees
total_trees = num_epochs * trees_per_epoch

print(f"\n=== Model Configuration ===")
print(f"Epochs: {num_epochs}")
print(f"Trees per epoch: {trees_per_epoch}")
print(f"Total trees: {total_trees}")
print(f"Recommended epochs: 10-15 for optimal performance")
print(f"Current setting will train {total_trees} trees total")
print("="*40)

# -------- Normalization --------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# -------- Transforms --------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# -------- Dataset and Dataloader --------
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                              shuffle=True, num_workers=2)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Classes:", class_names)

# -------- Extract TDA Features --------
def extract_all_tda_features(dataloader, device):
    """
    Extract TDA features from all images in the dataloader.
    """
    all_features = []
    all_labels = []
    
    print("Extracting TDA features...")
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        tda_features = extract_tda_features(inputs)
        all_features.append(tda_features.numpy())
        all_labels.extend(labels.numpy())
    
    # Concatenate all features
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    return X, y

# -------- Train and Evaluate Random Forest with History --------
def train_and_evaluate_rf_with_history(X_train, y_train, X_test, y_test, class_names, n_estimators=100):
    """
    Train Random Forest with TDA features, track history, and evaluate.
    """
    print("Training Random Forest with history tracking...")
    model = get_random_forest(n_estimators=n_estimators, max_depth=None, random_state=42)
    model, history = train_random_forest_with_history(model, X_train, y_train, X_test, y_test, n_estimators)
    
    print("Evaluating Random Forest...")
    results = evaluate_random_forest(model, X_test, y_test, class_names)
    
    return model, results, history

# -------- Plot and Save Curves --------
def plot_and_save_curves(train_scores, val_scores, out_dir):
    plt.figure()
    plt.plot(train_scores, label='Train Score')
    plt.plot(val_scores, label='Validation Score')
    plt.legend()
    plt.title('Random Forest Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.savefig(os.path.join(out_dir, "rf_training_curve.png"))
    plt.close()

# -------- Save Confusion Matrix and Report --------
def save_confusion_matrix_and_report(y_true, y_pred, class_names, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    # Save classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, labels=[i for i in range(len(class_names))]
    )
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

# -------- Feature Importance Visualization --------
def plot_feature_importance(model, out_dir):
    """
    Plot and save feature importance from Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("TDA Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(os.path.join(out_dir, "feature_importance.png"))
    plt.close()

# -------- Main Pipeline --------
print("Starting TDA + Random Forest Pipeline...")

# Extract TDA features from training data
print("Extracting TDA features from training data...")
X_train, y_train = extract_all_tda_features(dataloaders['train'], device)

# Extract TDA features from validation data
print("Extracting TDA features from validation data...")
X_val, y_val = extract_all_tda_features(dataloaders['val'], device)

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")

# Train and evaluate Random Forest with history tracking
model, results, history = train_and_evaluate_rf_with_history(X_train, y_train, X_val, y_val, class_names, n_estimators=total_trees)

# Print results
print(f"\nRandom Forest Results:")
print(f"Validation Accuracy: {results['accuracy']:.4f}")
print(f"\nClassification Report:")
print(results['classification_report'])

# Plot and save training history
print("\nPlotting training history...")
plot_training_history(history, "training_history.png")

# Save model
save_random_forest(model, "random_forest_model.pkl")

# Save confusion matrix and report
save_confusion_matrix_and_report(y_val, results['predictions'], class_names, ".")

# Plot feature importance
plot_feature_importance(model, ".")

# Plot validation accuracy as a bar plot
plt.figure()
plt.bar(['Validation Accuracy'], [results['accuracy']])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Random Forest Validation Accuracy')
plt.savefig('validation_accuracy.png')
plt.close()

print("\n" + "="*60)
print("FINAL TRAINING SUMMARY")
print("="*60)
print(f"Final Training Accuracy: {history['train_scores'][-1]:.4f} (↑ {history['train_scores'][-1]*100:.1f}%)")
print(f"Final Validation Accuracy: {history['val_scores'][-1]:.4f} (↑ {history['val_scores'][-1]*100:.1f}%)")
print(f"Final Training Loss: {history['train_losses'][-1]:.4f} (↓ {history['train_losses'][-1]*100:.1f}%)")
print(f"Final Validation Loss: {history['val_losses'][-1]:.4f} (↓ {history['val_losses'][-1]*100:.1f}%)")

# Show improvement from start to end
if len(history['train_scores']) > 1:
    total_train_improvement = history['train_scores'][-1] - history['train_scores'][0]
    total_val_improvement = history['val_scores'][-1] - history['val_scores'][0]
    print(f"\nTotal Training Accuracy Improvement: {total_train_improvement:+.4f}")
    print(f"Total Validation Accuracy Improvement: {total_val_improvement:+.4f}")

print("\nPipeline completed successfully!")
print(f"Model saved as: random_forest_model.pkl")
print(f"Training history plot saved as: training_history.png")
print(f"Single accuracy chart saved as: training_accuracy_progress.png")
print(f"Results saved in current directory")
