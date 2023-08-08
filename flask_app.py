from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model and tokenizer
model = load_model('models/spam_detection_model.h5')
with open('utils/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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

if __name__ == '__main__':
    app.run(debug=True)

 