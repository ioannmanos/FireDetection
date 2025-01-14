# FireDetection
This project focuses on developing a deep learning-based system for the early detection of forest fires using image data. Utilizing the VGG16 architecture, a pre-trained convolutional neural network (CNN), the system aims to classify images into 'fire' and 'non-fire' categories with high accuracy. The project pipeline includes data augmentation techniques such as rotation, zoom, brightness adjustment, and custom preprocessing steps like random cropping and cutout augmentation to enhance the robustness of the model.

Images are sourced from a structured dataset divided into training, validation, and testing sets. The VGG16 model is fine-tuned by freezing the initial layers and training additional dense layers with dropout and batch normalization for improved performance and regularization. Advanced callbacks like early stopping, learning rate reduction, and model checkpointing are employed to optimize the training process and avoid overfitting.

The system's performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curves, alongside visualizations of the confusion matrix. Additionally, a dedicated script allows for testing the model on new images, providing predictions and confidence scores.

Overall, this project aims to leverage state-of-the-art deep learning techniques to create a reliable tool for forest fire detection, potentially aiding in timely interventions and disaster management efforts.

Keywords: Fire detection , Deep learning , Convolutional Neural Network (CNN) , VGG16
