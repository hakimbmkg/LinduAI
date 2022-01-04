# LinduAI

Python Package for Station Quality Analysis and Predicted Magnitude Based on Convolutional Neural Network

<img width="1076" alt="Screen Shot 2022-01-03 at 15 14 12" src="https://user-images.githubusercontent.com/28749749/147910340-bb900def-0324-442d-921d-08aae3f75948.png">

# <b>Installation</b>

```
python setup.py install
```

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

# Requirement
- python 3.8
- tensorflow 2.4.1
- numpy
- obspy
- keras
- librosa

# Example

```
from LinduAI.main.model import Models

Models.testmodels('/your/path/mseed_waveform')

```

