import os
from setuptools import setup
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name='LinduAI',
	version='0.0.1',
	author='hakimBMKG',
	author_email='arif.hakim@bmkg.go.id',
	description='Python Package for Station Quality Analysis and Predicted Magnitude Based on Convolutional Neural Network',
	license='MIT License',
	keywords='Station Earthquake Quality Magnitude Prediction CNN ',
	url='https://github.com/hakimbmkg/LinduAI',
	long_description=read('README.md'),
	packages=['LinduAI'],
	install_requires=[
	'pytest',
	'numpy==1.19.5',
	'pkginfo',
	'scipy==1.7.1', 
	'tensorflow==2.4.1', 
	'keras', 
	'matplotlib', 
	'pandas',
	'tqdm==4.62.3', 
	'h5py', 
	'obspy',
	'jupyter'], 

    python_requires='>=3.6',
	)