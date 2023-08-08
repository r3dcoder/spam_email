import pkg_resources

from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Define the path to the model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), 'models', 'spam_detection_model.h5')
tokenizer_path = os.path.join(os.path.dirname(__file__), 'utils', 'tokenizer.pickle')


# Load the pre-trained model and tokenizer
model = load_model(model_path)
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

app = Flask(__name__)

# Define a route to the form page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            print("email: "+email)
            # Tokenize and pad the email
            sequences = tokenizer.texts_to_sequences([email])

            padded_sequences = pad_sequences(sequences, maxlen=100)

            # Make a prediction using the model
            prediction = model.predict(padded_sequences)
            spam_probability = prediction[0][0]
            #spam_probability = prediction[0][0]
            #spam_probability = prediction[0][0]

            # Classify the email as spam or not spam based on the probability threshold..
            threshold = 0.5
            if spam_probability > threshold:
                classification = "Not Spam"
            else:
                classification = "spam"
            print(classification)

            return render_template('result.html', email=email, classification=classification)

        except Exception as e:
            # Handle the exception and return an error page
            return render_template('error.html', error=str(e))

    return render_template('form.html')



def get_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not installed"

 
# Example usage
print(f"Python version: {get_version('python')}")
print(f"TensorFlow version: {get_version('tensorflow')}")
print(f"Keras version: {get_version('keras')}")
print(f"numpy version: {get_version('numpy')}")
print(f"pandas version: {get_version('pandas')}")
print(f"Flask version: {get_version('Flask')}")
print(f"scikit-learn version: {get_version('scikit-learn')}")

if __name__ == '__main__':
    app.run(debug=True)

 