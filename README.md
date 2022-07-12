# LinduAI

<a href="https://doi.org/10.5281/zenodo.6785556"><img width="200" alt="Zenodo-DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.6785556.svg"/></a>

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
git clone https://github.com/hakimbmkg/LinduAI
```

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
* Mseed Event also in folder example

```

Cite as
```
Hakim, Arif Rachman. (2022). LinduAI (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.6785556
```
```
@software{hakim_arif_rachman_2022_6785556,
  author       = {Hakim, Arif Rachman},
  title        = {LinduAI},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6785556},
  url          = {https://doi.org/10.5281/zenodo.6785556}
}
```
