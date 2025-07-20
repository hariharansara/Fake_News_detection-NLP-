#!/usr/bin/env bash
# exit on error
set -o errexit

# Install the dependencies from requirements.txt
pip install -r requirements.txt

# Run a Python script to download the NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
