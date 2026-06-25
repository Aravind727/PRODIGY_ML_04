import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

dataset_path = "dataset"

data = []
labels = []

print("Loading images...")

for image_name in os.listdir(dataset_path):

    image_path = os.path.join(dataset_path, image_name)

    img = cv2.imread(image_path)

    if img is None:
        continue

    try:
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        label = image_name.split("_")[2]

        data.append(img.flatten())
        labels.append(label)

    except:
        pass

X = np.array(data)
y = np.array(labels)

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM Model...")

svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12,5))

for i in range(5):

    plt.subplot(1,5,i+1)

    img = X_test[i].reshape(64,64)

    plt.imshow(img, cmap="gray")

    plt.title(f"Pred: {y_pred[i]}")

    plt.axis("off")

plt.tight_layout()

plt.savefig("gesture_predictions.png")

plt.show()

print("\nOutput saved as gesture_predictions.png")