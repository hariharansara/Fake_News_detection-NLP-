#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Create a directory for the NLTK data and download the required packages to it
python -c "import nltk; nltk.download('stopwords', download_dir='./nltk_data'); nltk.download('punkt', download_dir='./nltk_data'); nltk.download('wordnet', download_dir='./nltk_data')"
