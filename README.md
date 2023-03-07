# An Artwork Search Engine for the Metropolitan Museum of Art

## Table of Contents
* [Abstract](#Abstract)
* [Installation](#Installation)
* [Usage](#Usage)
* [Licensing](#Licensing)

## Abstract
The Metropolitan Museum of Art (MET) is a world-renowned museum located in New York City that features a diverse collection of artworks spanning over 5,000 years. Despite its reputation, the MET’s current search engine has limited capabilities and is not user-friendly, making it difficult for visitors to find specific artworks or information about them. To address this issue, our project develops a new search engine interface for the MET that utilizes advanced information retrieval machine learning algorithms and natural language processing techniques, as well as additional features such as cross-language search, query expansion, misspelling correction, and ranking by relevance. We use artworks’ metadata and introductory description provided by MET to build the IR system using learning-to-rank techniques and develop a web interface using Flask to allow users to interact with the search engine and receive the top 5 relevant results. Comprehensive model evaluation has shown that our efforts have significantly improved the accuracy and efficiency of the baseline model.

The demonstration video can be accessed via the following [link](https://drive.google.com/file/d/19MU9Lf--nBWypDeRCnyjkP-r0eB0z7nz/view?usp=share_link).

## Installation
This code requires several Python packages to be installed: python-terrier, fastrank, lightgbm, googletrans, swig, jamspell, selenium, pandas, numpy, nltk, beautifulsoup4, xgboost, sklearn, pyterrier.

You can install the required packages by running the following commands:

<pre><code>!pip install python-terrier
!pip install fastrank
!pip install lightgbm
!pip install googletrans==4.0.0rc1
!sudo apt-get install swig
!sudo pip install jamspell
!wget https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
!tar -xvf en.tar.gz
!pip install -U selenium
!apt update
!apt install chromium-chromedriver
</code></pre>

## Usage
To access the data used in this project, please use the [link](https://drive.google.com/drive/folders/1eScftscDYLhtz_88KxEezMDtkD5RU2K8?usp=sharing) to download the dataset from Google Drive.

To start Flask web application that allows users to interact with search engine.
1. Navigate to the main directory of the Interface folder in this project.
2. Run `flask run` to start the Flask app:
3. Wait until the model training is complete.
4. You can access the application by navigating to http://localhost:5000 in your web browser.

## Licensing
This project is licensed under the MIT License - see the LICENSE file for details.

Authors: Ruqin Chang (ruqinch@umich.edu), Wenjie Wu (wuwenj@umich.edu)
