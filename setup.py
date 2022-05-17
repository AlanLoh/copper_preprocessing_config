#! /usr/bin/python3
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


setup(
    name="copperconfig",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "astropy",
    ],
    python_requires=">=3.6",
    scripts=[
        "bin/parset2preprocconfig"
    ],
    version="0.1",
    description='NenuFAR Parset to COPPER configuration file converter',
    url="https://github.com/AlanLoh/copper_preprocessing_config.git",
    author="Alan Loh",
    author_email="alan.log@obspm.fr",
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research"
    ],
    zip_safe=False
)

