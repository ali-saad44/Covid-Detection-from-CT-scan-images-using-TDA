from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
import matplotlib.pyplot as plt

def get_random_forest(n_estimators=100, max_depth=None, random_state=42):
    """
    Returns a Random Forest classifier for TDA features.
    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed for reproducibility
    Returns:
        RandomForestClassifier: Configured Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    return model

def train_random_forest_with_history(model, X_train, y_train, X_val, y_val, n_estimators=100):
    """
    Train the Random Forest model with TDA features and track training history.
    Args:
        model: RandomForestClassifier instance
        X_train: Training TDA features
        y_train: Training labels
        X_val: Validation TDA features
        y_val: Validation labels
        n_estimators: Number of estimators to train
    Returns:
        tuple: (trained_model, training_history)
    """
    train_scores = []
    val_scores = []
    train_losses = []
    val_losses = []
    
    # Calculate trees per epoch (default 10)
    trees_per_epoch = 10
    if n_estimators > 100:
        trees_per_epoch = 15  # For larger models, use more trees per epoch
    
    print("\n=== Starting Random Forest Training with Epoch-like Progress ===")
    print(f"Training will show progress after each batch of {trees_per_epoch} trees...")
    print(f"Total epochs: {n_estimators // trees_per_epoch}")
    
    # Train with increasing number of estimators to track progress
    for i in range(trees_per_epoch, n_estimators + 1, trees_per_epoch):
        # Create a new model with current number of estimators
        temp_model = RandomForestClassifier(
            n_estimators=i,
            max_depth=model.max_depth,
            random_state=model.random_state,
            n_jobs=-1
        )
        
        # Train the model
        temp_model.fit(X_train, y_train)
        
        # Calculate scores
        train_score = temp_model.score(X_train, y_train)
        val_score = temp_model.score(X_val, y_val)
        
        # Calculate losses (1 - accuracy for classification)
        train_loss = 1 - train_score
        val_loss = 1 - val_score
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress after each "epoch"
        epoch_num = i // trees_per_epoch
        print(f"\n--- Epoch {epoch_num} (Trees: {i}) ---")
        print(f"Training Accuracy: {train_score:.4f} (↑ {train_score*100:.1f}%)")
        print(f"Validation Accuracy: {val_score:.4f} (↑ {val_score*100:.1f}%)")
        print(f"Training Loss: {train_loss:.4f} (↓ {train_loss*100:.1f}%)")
        print(f"Validation Loss: {val_loss:.4f} (↓ {val_loss*100:.1f}%)")
        
        # Show improvement from previous epoch
        if len(train_scores) > 1:
            train_improvement = train_scores[-1] - train_scores[-2]
            val_improvement = val_scores[-1] - val_scores[-2]
            print(f"Training Accuracy Improvement: {train_improvement:+.4f}")
            print(f"Validation Accuracy Improvement: {val_improvement:+.4f}")
        
        # Plot current progress
        plot_current_progress(train_scores, val_scores, train_losses, val_losses, epoch_num)
    
    # Train the final model with all estimators
    model.fit(X_train, y_train)
    
    history = {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'estimators_range': list(range(trees_per_epoch, n_estimators + 1, trees_per_epoch))
    }
    
    return model, history

def plot_current_progress(train_scores, val_scores, train_losses, val_losses, epoch_num):
    """
    Plot current training progress after each epoch - simple line chart like the attached image.
    """
    plt.figure(figsize=(10, 6))
    
    # Create epochs list (starting from 0 like the image)
    epochs = list(range(len(train_scores)))
    
    # Plot accuracy lines exactly like the attached image
    plt.plot(epochs, train_scores, 'r-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_scores, 'b-', label='Validation Accuracy', linewidth=2)
    
    # Set up the plot like the image
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy after epoch: {epoch_num}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis range like the image (0.80 to 1.00)
    plt.ylim(0.80, 1.00)
    
    # Set x-axis ticks like the image
    if len(epochs) > 20:
        plt.xticks(range(0, len(epochs), 20))
    else:
        plt.xticks(epochs)
    
    # Set y-axis ticks like the image
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    
    plt.tight_layout()
    plt.savefig('training_accuracy_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Updated accuracy chart saved: training_accuracy_progress.png")

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot final training and validation accuracy curves like the attached image.
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create epochs list (starting from 0 like the image)
    epochs = list(range(len(history['train_scores'])))
    
    # Plot accuracy lines exactly like the attached image
    plt.plot(epochs, history['train_scores'], 'r-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, history['val_scores'], 'b-', label='Validation Accuracy', linewidth=2)
    
    # Set up the plot like the image
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy after epoch: {len(epochs)}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis range like the image (0.80 to 1.00)
    plt.ylim(0.80, 1.00)
    
    # Set x-axis ticks like the image
    if len(epochs) > 20:
        plt.xticks(range(0, len(epochs), 20))
    else:
        plt.xticks(epochs)
    
    # Set y-axis ticks like the image
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final training history plot saved as: {save_path}")

def train_random_forest(model, X_train, y_train):
    """
    Train the Random Forest model with TDA features.
    Args:
        model: RandomForestClassifier instance
        X_train: Training TDA features
        y_train: Training labels
    Returns:
        RandomForestClassifier: Trained model
    """
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X_test):
    """
    Make predictions using the trained Random Forest.
    Args:
        model: Trained RandomForestClassifier
        X_test: Test TDA features
    Returns:
        tuple: (predictions, prediction_probabilities)
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return predictions, probabilities

def evaluate_random_forest(model, X_test, y_test, class_names):
    """
    Evaluate the Random Forest model.
    Args:
        model: Trained RandomForestClassifier
        X_test: Test TDA features
        y_test: True labels
        class_names: List of class names
    Returns:
        dict: Evaluation metrics
    """
    predictions, probabilities = predict_random_forest(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'classification_report': classification_report(y_test, predictions, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }
    
    return results

def save_random_forest(model, filepath):
    """
    Save the trained Random Forest model.
    Args:
        model: Trained RandomForestClassifier
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)

def load_random_forest(filepath):
    """
    Load a trained Random Forest model.
    Args:
        filepath: Path to the saved model
    Returns:
        RandomForestClassifier: Loaded model
    """
    return joblib.load(filepath) 