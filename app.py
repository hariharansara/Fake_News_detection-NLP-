# --- 1. Import Necessary Libraries ---
from flask import Flask, request, render_template
import joblib
import re
import nltk
import os
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os # To handle file paths robustly

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Load The Trained Model and Vectorizer ---
# We load the model and vectorizer once when the app starts to be efficient.

# Construct absolute paths to ensure files are found regardless of where the script is run
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

try:
    print("Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    print("Please ensure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the same directory as app.py.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"An error occurred while loading the files: {e}")
    model = None
    vectorizer = None


# --- 4. Define the Text Preprocessing Function ---
# This MUST be the same function used during the model training phase.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_process(text):
    """
    Cleans and preprocesses raw text by:
    1. Converting to lowercase.
    2. Removing punctuation and special characters.
    3. Tokenizing the text.
    4. Removing stop words.
    5. Lemmatizing the tokens.
    6. Joining tokens back into a string.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs
        
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    
    tokens = word_tokenize(text)  # Tokenize the text
    
    # Remove stop words and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(processed_tokens) # Join tokens back into a single string

# --- 5. Define Flask Application Routes ---

# Route for the home page (displays the input form)
@app.route('/')
def home():
    """Renders the main page (index.html)."""
    return render_template('index.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission, processes the input, makes a prediction,
    and renders the result on the home page.
    """
    if request.method == 'POST':
        # Get data from the form
        news_title = request.form.get('title', '')
        news_text = request.form.get('text', '')

        if not news_title and not news_text:
            return render_template('index.html', prediction_text="Please enter a title or article text.", prediction_class="fake")

        # Combine title and text for a full analysis
        full_text = news_title + " " + news_text
        
        # Preprocess the text
        processed_text = clean_and_process(full_text)
        
        # Use the loaded vectorizer to transform the text
        # Note: We use .transform(), not .fit_transform()
        vectorized_text = vectorizer.transform([processed_text])
        
        # Use the loaded model to make the prediction
        prediction = model.predict(vectorized_text)
        
        # Determine the result and the corresponding CSS class for styling
        if prediction[0] == 1:
            result_text = "This appears to be FAKE news."
            result_class = "fake"  # Corresponds to red color in CSS
        else:
            result_text = "This appears to be REAL news."
            result_class = "real" # Corresponds to green color in CSS
            
        # Render the page again, but this time with the prediction results
        return render_template('index.html', 
                               prediction_text=result_text, 
                               prediction_class=result_class)

# --- 6. Run the Flask App ---
if __name__ == '__main__':
    # Download necessary NLTK data if it's not already present
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK data (stopwords, punkt, wordnet)...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK data downloaded.")

    # Run the app in debug mode for development
    # For production, you would use a proper web server like Gunicorn
    app.run(debug=True)
