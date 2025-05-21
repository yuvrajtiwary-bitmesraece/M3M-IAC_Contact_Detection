# M3M-IAC_Contact_Detection
Two-stage dental X-ray analysis pipeline: DeepLabV3+ segments teeth &amp; nerve regions; MobileNet classifies contact vs no-contact using Ground and Predicted masks. Includes preprocessing, training, evaluation, and visualization. Useful for dental AI and medical imaging research.

## Project Pipeline Overview

1. **Data Extraction & Loading**  
   Raw dental X-ray images and masks are extracted and loaded for preprocessing.

2. **Preprocessing & Mask Conversion**  
   Images resized and normalized; grayscale masks converted into multi-class masks (teeth, nerve, overlap).

3. **Segmentation Model (DeepLabV3+) Training**  
   Train DeepLabV3+ with MobileNetV2 backbone to segment teeth and nerve regions accurately.

4. **Segmentation Output Visualization**  
   Visualize input images, ground truth masks, and predicted segmentation masks side-by-side.

5. **Classification Dataset Preparation**  
   Create labels for contact/no-contact classes from the segmentation masks.

6. **Classification Model (MobileNet) Training**  
   Fine-tune MobileNet on original and segmentation masks to classify contact vs no-contact.

7. **Evaluation & Visualization**  
   Evaluate model performance with metrics, confusion matrix, and prediction visualization.
