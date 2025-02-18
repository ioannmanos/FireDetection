# FireDetection
This thesis focuses on developing a fire detection system leveraging deep learning techniques and image data. Specifically, the VGG16 architecture, a pre-trained convolutional neural network (CNN), is employed and fine-tuned to classify images into two categories: "fire" and "non-fire." To enhance the model's generalization capability, various data augmentation techniques and custom preprocessing functions are applied.

The system is trained and evaluated on a structured dataset, divided into training, validation, and test datasets. During training, the initial layers of the VGG16 model remain frozen, while additional fully connected layers are introduced with normalization, batch normalization and dropout techniques to improve performance. Furthermore, advanced methods, including early stopping, learning rate scheduling, and model checkpointing, are applied to prevent overfitting.

The evaluation of the system's performance is primarily conducted using metrics such as accuracy and loss. Additionally, metrics such as precision, recall, F1-score (harmonic mean of precision and recall), and AUC-ROC curves are utilized, while the performance is also visualized through the normalized confusion matrix. Finally, a testing mechanism is developed to evaluate the model on new images, providing predictions along with corresponding confidence scores.

Overall, this work explores and implements state-of-the-art deep learning techniques to develop a reliable fire detection system, with the potential to contribute to timely fire detection and disaster management efforts.

Keywords: Fire detection , Deep learning , Convolutional Neural Network (CNN) , VGG16

