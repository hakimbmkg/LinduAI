# LinduAI

Python Package for Station Quality Analysis and Predicted Magnitude Based on Convolutional Neural Network

<img width="1077" alt="Screen Shot 2022-01-04 at 15 31 46" src="https://user-images.githubusercontent.com/28749749/148031066-866757d9-c696-4ece-8808-fff635a52178.png">

# <b>Installation</b>

Using Conda, make new env conda (*recommended)
```
conda create -n linduai python=3.8
```
```
pip install tensorflow==2.4.1
```
```
pip install LinduAI
```
if pip failed to install LinduAI, you can manual install with 

```git clone https://github.com/hakimbmkg/LinduAI```

```
python setup.py install
```

# Requirement
- python 3.8
- tensorflow 2.4.1
- numpy
- obspy
- keras
- librosa

# Example

```
from LinduAI.main.modelmag import Modelsmag

Modelsmag.predictedmag('/your/path/CISI.IA_20090616044802.645_EV', 'input/your/models_mags')


* you can used model in folder example (LinduAI_Models)

```

