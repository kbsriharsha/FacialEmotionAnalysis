# FacialEmotionAnalysis ![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000) ![](https://img.shields.io/badge/Harsha-Karpurapu-brightgreen.svg?colorB=ff0000)

This program helps in identifying a person's emotion from live stream. The model inspiration was taken a research paper, where the authors published a efficient network for facial and gender identification (https://arxiv.org/pdf/1710.07557.pdf)

### Code Requirements
Just execute the below command from your terminal for installing all the required dependencies. 
##### pip install requirement.txt

### Model Description

This program uses the network named exception (https://arxiv.org/pdf/1710.07557.pdf), which is a combination of several residual blocks. I trained the network for 100 epochs which took me 56 hours on the MAC with 16gb ram. However I strongly recommend training your model on Google Colab using GPU's, where it took me 45 minutes. `56hours to 45 min :D`

### Procedure for using the code
- For training your model, run `python training.py`. However I strongly recommend using model I trained it for you. 
- For using the model on a live stream, run `python RealtimeAnalysis.py` where you can capture the live stream from the webcam and the model inference will be displayed. 
- Notes: Use the gpu version of the model for the trained model

### References
- Original Paper: https://arxiv.org/pdf/1710.07557.pdf
- Program Implementation of the published paper: https://github.com/oarriaga/face_classification
- Keras: https://keras.io/
