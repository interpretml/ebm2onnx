#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'onnx>=1.8',
    'interpret-core[required,ebm]>=0.3',
 ]

setup(
    author="Romain Picard",
    author_email='romain.picard@softathome.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="EBM model serialization to ONNX",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='ebm2onnx',
    name='ebm2onnx',
    packages=find_packages(include=['ebm2onnx']),
    url='https://github.com/interpretml/ebm2onnx.git',
    version='2.0.0',
    zip_safe=True,
)
