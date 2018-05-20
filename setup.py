'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='pyEye',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Tom Gonda awesome ML platform',
      author='Tom Gonda',
      author_email='tom.gonda@gmail.com',
      license='Unlicense',
      install_requires=[
            'keras',
            'h5py',
            'statistics',
            'scikit-image',
            'psutil',
            'face_recognition'],
      zip_safe=False)