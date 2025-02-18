import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Φόρτωση των εικόνων από τον φάκελο
datagen = ImageDataGenerator(rescale=1./255)  # Κανονικοποίηση των τιμών των pixel (0-1)

test_generator = datagen.flow_from_directory(
    'C:/Users/user/source/repos/FireDetection/Training',  # Φάκελος με τις εικόνες
    target_size=(224, 224),  # Προσαρμογή στο μέγεθος που δέχεται το μοντέλο
    batch_size=32,  # Αριθμός εικόνων ανά batch
    class_mode="binary",  # Δυαδική ταξινόμηση (0 ή 1)
    shuffle=False  # Χωρίς shuffle για σταθερές προβλέψεις
)

# 2. Φόρτωση του προεκπαιδευμένου μοντέλου
model = tf.keras.models.load_model("C:/Users/user/source/repos/FireDetection/VGG_finetuned_model.h5")  # Προσαρμόστε το path αν χρειάζεται

# 3. Υπολογισμός απώλειας και ακρίβειας στο test set
loss, accuracy = model.evaluate(test_generator)

# 4. Πρόβλεψη των αποτελεσμάτων
y_pred_probs = model.predict(test_generator)  # Προβλέπει πιθανότητες [0,1]

# 5. Μετατροπή πιθανοτήτων σε δυαδικές κατηγορίες (0 ή 1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  

# 6. Απόκτηση των πραγματικών ετικετών
y_true = test_generator.classes  # Οι πραγματικές κατηγορίες από το generator

# Εκτύπωση δεδομένων για debugging
print("Original y_true:", y_true[:10])
print("First 5 probabilities:", y_pred_probs[:5].flatten())
print("Converted y_pred:", y_pred[:10])

# Εκτύπωση όλων των labels για επιβεβαίωση
print("All true labels:", y_true)

# 7. Υπολογισμός μετρικών
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 8. Εκτύπωση των αποτελεσμάτων
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 9. Κανονικοποιημένος Confusion Matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')

# 10. Εμφάνιση του confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fire", "Non-Fire"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.show()

# Διάγραμμα Precision, Recall, F1 Score
plt.figure(figsize=(10, 6))
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'orange', 'purple'])
plt.title('Metrics: Precision, Recall, F1 Score')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.show()

# Εκτύπωση αριθμού εικόνων για επιβεβαίωση
print("Total test images (generator):", test_generator.samples)
