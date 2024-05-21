from flask import Flask, request, jsonify
import neattext.functions as nfx
import joblib
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


app = Flask(__name__)

def load_files():
    global model, tfidf_vectorizer
    # Load the model
    model = joblib.load('naive_bayes_model.pkl')

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load files only once during app initialization
load_files()

def clean_text(text):
    cleaned_text = nfx.remove_special_characters(text)  # Remove special characters
    cleaned_text = nfx.remove_stopwords(text)  # Remove stopwords
    # Additional preprocessing steps can be added here
    return cleaned_text


def send_email(user_details):
    # Email configuration
    sender_email = "lifesaver102023@gmail.com"
    recipient_list = ['svspbs567@gmail.com']

    # Create message container - the correct MIME type is multipart/alternative
    msg = MIMEMultipart()
    msg['Subject'] = f"Urgent Concern for {user_details['name']}'s Well-being"
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_list)

    # Create the body of the message
    message = f'''Dear Mental Health Support Foundation Team,
    
We have identified a post on our platform that raises serious concerns about the user's mental health and safety. We are reaching out to seek your immediate intervention and support.

User Details:
Name: {user_details['name']}
Mobile No.: {user_details['mobile_number']}
Email: {user_details['email']}

Given the gravity of the situation, we kindly request your assistance in connecting with the user to ensure their safety and well-being. Your expertise and resources could be invaluable in providing the necessary support and guidance during this critical time.

Thank you for your prompt attention to this matter.

Sincerely,
MindCare Support Team'''

    # Attach the message to the email
    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, 'aouojaqtpwhlzoiy')
        server.sendmail(sender_email, recipient_list, msg.as_string())

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        texts = data.get('texts', [])
        predictions = []
        predictions_score = []

        for text in texts:
            cleaned_text = clean_text(text)
            X_test = tfidf_vectorizer.transform([cleaned_text])
            y_probs = model.predict_proba(X_test)[:, 1]
            predictions.append(y_probs[0])
            predictions_score.append(y_probs[0])

        # Convert numpy arrays to lists and ensure they are serializable
        predictions = [float(pred) for pred in predictions]
        predictions_score = [float(score) for score in predictions_score]
        
        if predictions[0] > 0.9:
            
            user_details = {
                'name': data.get('user_name', []),  # Replace with user['user_name']
                'mobile_number': data.get('mobile_number', []),  # Replace with user['mobile_no']
                'email': data.get('email', [])  # Replace with email
            }
            send_email(user_details)

        return jsonify({"predicted_labels": predictions, "predicted_scores": predictions_score})

if __name__ == '__main__':
    app.run(debug=False)
