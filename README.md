# AAI-501-BaxterFarag-FinalProject

READ ME

Convolutional Neural Networks and Random Forest Classifiers For Enhanced Breast Cancer Diagnosis 
github: https://github.com/jbaxterusd/AAI-501-BaxterFarag-FinalProject

This project is part of the AAI-501 Course in the Applied Artificial Intelligence Program at the University of San Diego (USD)
--Project Status: [Completed]

Dataset: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)

Intro/Objective: 
The purpose of our project was to create a hybrid AI model to predict breast cancer diagnosis. Specifically, we aim to utilize image based classification using Convolutional Neural Networks and further analyze/predict the feature data with a Random Forest classifier. 

The scope of our project focuses on predicting the diagnosis of tumors (benign/malignant) based on significant mass segmentation of the tumor. 

Partners/Contributors: 
- Robair Farag (rfarag@sandiego.edu)
- Jack Baxter (8585314334, jackbaxter@sandiego.edu)

Methods Used: 
- ResNet50 (CNN) 
- Random Forest
- Machine Learning
- Data Visualization 
- Data Preprocessing / Cleaning

Technologies: 
- Python
- Jupyter Notebook
- Tensorflow
- Pydicom
- Numpy
- Sklearn
- Matplotlib 

Project Description: 
This project implements advanced artificial intelligence techniques to address critical challenges in breast cancer diagnosis, utilizing the **Curated Breast Imaging Subset of the Digital Database for Screening Mammography (CBIS-DDSM)**. The primary objective was to develop and evaluate two machine learning models: a **Convolutional Neural Network (CNN)** based on ResNet50 architecture and a **Random Forest Classifier**, to classify mammograms as benign or malignant with a focus on mass segmentation.

Data preprocessing involved resizing DICOM mammogram images to 224x224 grayscale format and converting them into NumPy arrays, ensuring compatibility with the ResNet50 model. A pre-trained ResNet50, fine-tuned on the CBIS-DDSM dataset, achieved an accuracy of 61.11%. Its residual connections helped mitigate vanishing gradient issues and enhance feature extraction for mass segmentation. Additionally, a Random Forest Classifier was implemented using features extracted from the penultimate layer of ResNet50. This approach leveraged the interpretability and noise-reduction strengths of Random Forests, achieving an accuracy of 48.41%. Analysis of results revealed areas for optimization, particularly in preprocessing and model design, given the equal class distribution (benign vs. malignant) in the dataset.

Future directions include incorporating Region of Interest (ROI) masks from CBIS-DDSM to improve model specificity and align training with expert annotations. Enhanced hyperparameter tuning, such as enabling deeper layer training in ResNet50, could further boost performance. Improved data augmentation strategies and exploration of hybrid AI models are also proposed to enhance both accuracy and interpretability. These developments aim to bridge the gap between AI research and clinical application by addressing key barriers to adoption, such as model transparency and reliability.

This project demonstrates the potential of hybrid AI approaches in medical image classification while identifying opportunities for optimization and further research. It contributes to bridging the gap between AI and clinical practice by emphasizing interpretable and actionable insights for healthcare professionals.  

Aknowledgements: 

Thank you to our professor, Dave Friesen, for his continued support and guidance. Thank you to the MS AAI program at USD for the opportunity and curriculum to perform the project. 
