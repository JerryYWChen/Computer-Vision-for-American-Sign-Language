import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def load_data(data_dir):
    images = []
    labels = []
    class_names = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            class_names.append(folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    images.append(img)
                    labels.append(class_names.index(folder))
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names

def optimize_svm_random(X_train, y_train):
    print("Starting hyperparameter tuning using RandomizedSearchCV...")

    param_dist = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    random_search = RandomizedSearchCV(
        SVC(),
        param_distributions=param_dist,
        n_iter=50,
        scoring='accuracy',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("Best Parameters Found:", random_search.best_params_)
    print("Best Cross-Validation Score:", random_search.best_score_)

    return random_search.best_estimator_


def main():
    data_dir = "./archive/asl_alphabet_train/asl_alphabet_train" # change direction
    print("Loading data...")
    images, labels, class_names = load_data(data_dir)

    images = images.reshape(images.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, train_size=0.8, random_state=42)

    print("Optimizing SVM model...")
    classifier = optimize_svm_random(X_train, y_train)

    print("Evaluating the best model...")
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()


