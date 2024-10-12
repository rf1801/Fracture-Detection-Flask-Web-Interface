import base64
from flask import Flask, jsonify, redirect, request, render_template, url_for, session
import openai
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from flask_mail import Mail, Message
import pdfcrowd
import numpy as np
import os
import pandas as pd
import base64
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Average
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input



app = Flask(__name__)
app.secret_key = 'your_very_secret_key'  # Set a secret key for session management
app.config['UPLOAD_FOLDER'] = 'static/imagetest/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


global_path="models/"
model_files = {
    "Elbow": ["Mobilenetmodel.keras"],
    "Finger":  ["finetuned_DenseNet_XR_FINGER.keras","finetuned_ResNet_XR_FINGER.keras", "finetuned_VGG_XR_FINGER.keras" ],
    "Forearm": ["Mobilenetmodel.keras"],
    "Wrist":["finetuned_DenseNet_XR_WRIST.keras"],
    "Humerus": ["Mobilenetmodel.keras"],
    "Shoulder": ["finetuned_ResNet_XR_SHOULDER.keras","finetuned_DenseNet_XR_SHOULDER.keras","finetuned_VGG_XR_SHOULDER.keras"]
}


def preprocess_image(image_path, model_name , target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img_array = np.expand_dims(img, axis=0)
    if "Dense" in model_name:
        img_array = densenet_preprocess_input(img_array)
    elif "Res" in model_name:
        img_array = resnet50_preprocess_input(img_array)
    elif "VGG" in model_name:
        img_array = vgg16_preprocess_input(img_array)
    return img_array


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.precision_fn = tf.keras.metrics.Precision()
        self.recall_fn = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 * ((p * r) / (p + r + 1e-10)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587 
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = ''
app.config['MAIL_PASSWORD'] = ''

mail = Mail(app)
UPLOAD_FOLDER = os.path.abspath("static/imagetest")

# Set the absolute path to the directory where images are stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# pdfcrowd client setup
api_username = ''  # Replace with your pdfcrowd username
api_key = ''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def extract_file_name(file_path): 
 
    base_name = os.path.basename(file_path) 
    return os.path.splitext(base_name)[0] 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '')
            surname = request.form.get('surname', '')
            dob = request.form.get('dob', '')
            email = request.form.get('email', '')
            file = request.files['inputpic']
            body_part = request.form.get('body_part')

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)






                models = model_files.get(body_part)
                predictions = []

                for model_file in models:
                    img_array = preprocess_image(image_path, model_file)
                    model = load_model(global_path+ model_file, custom_objects={'F1Score': F1Score})
                    output = model.predict(img_array)
                    predictions.append(output[0][0])
                pred_avg = np.array(Average()(predictions))



                prob_fractured = pred_avg#(pred_avg > 0.5).astype(int)
                prob_not_fractured = (1 - prob_fractured)
                if prob_fractured >= 0.5:
                    prediction = "Fractured"
                else:
                    prediction = "Not Fractured"


                image_url = url_for('static', filename=f'imagetest/{filename}')

                # Store data in session
                session['data'] = {
                    'name': name,
                    'surname': surname,
                    'dob': dob,
                    'email': email,
                    'prediction': prediction,
                    'prob_fractured': prob_fractured * 100,
                    'prob_not_fractured': prob_not_fractured * 100,
                    'image_path': image_url,
                    'body_part': body_part,
                    'file_name': extract_file_name(image_url) 

                }

                # Generate and send email with the PDF
                email_status = send_email()

                result_content = render_template("result_content.html", **session['data'])
                return render_template("index.html", result_content=result_content, email_status=email_status, **session['data'])
            else:
                raise ValueError("The file type is not allowed, use png, jpg, jpeg  only.")
        except Exception as e:
            app.logger.error(f"Failed to process request: {str(e)}")
            return render_template("index.html", error=f"Error: {str(e)}")

    data = session.get('data', {'prediction': "No prediction yet"})
    return render_template("index.html", **data)

@app.route('/pdfgen')
def pdfgen():
    if 'data' in session:
        data = session['data']
        return render_template("pdfgen.html", **data)
    else:
        # Redirect to home if no data to display
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()  # Clear all data stored in session
    return redirect(url_for('index'))


@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        pdfdata = session.get('data')
        if not pdfdata:
            raise ValueError("Session data 'data' not found.")

        recipient_email = pdfdata.get('email')
        if not recipient_email:
            raise ValueError("Recipient email not found in session data.")

        # Render the HTML template with session data
        html_content = render_template('pdfgen.html', **pdfdata)

        # Read the CSS file content
        css_path = os.path.join(app.root_path, 'static/css/styles.css')
        with open(css_path, 'r') as css_file:
            css_content = css_file.read()

        # Embed CSS in the HTML content
        html_content = f'<style>{css_content}</style>{html_content}'

        # Check and get the image path
        image_path = pdfdata.get('image_path')
        print("imaeg opath :", image_path)

        if image_path:
            # Construct the relative path to the image
            image_abs_path = os.path.join('C:\\Users\\Rednaks\\Documents\\abpdfbackend', 'static\\imagetest\\1-rotated2-rotated3-rotated3.jpg')
            if not os.path.exists(image_abs_path):
                raise FileNotFoundError(f"Image file not found: {image_abs_path}")

            with open(image_abs_path, 'rb') as image_file:
                image_data = image_file.read()
    
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            print("Image Base64 Data:", image_base64[:50])  # Debugging


        # Convert HTML to PDF using pdfcrowd API
        client = pdfcrowd.HtmlToPdfClient(api_username, api_key)
        pdf_file = client.convertString(html_content)

        # Attach the generated PDF to the email
        msg = Message(
            subject="Static Prediction Results",
            sender=app.config['MAIL_USERNAME'],
            recipients=[recipient_email]
        )
        msg.attach("result.pdf", "application/pdf", pdf_file)

        # Send the email
        mail.send(msg)

        return "Email sent successfully to [recipient_email] "
    except pdfcrowd.Error as e:
        app.logger.error(f"Pdfcrowd Error: {str(e)}")
        return f"Failed to generate or attach PDF: {str(e)}", 500
    except FileNotFoundError as e:
        app.logger.error(f"File not found error: {str(e)}")
        return f"File not found error: {str(e)}", 500
    except ValueError as e:
        app.logger.error(f"Session data error: {str(e)}")
        return f"Session data error: {str(e)}", 500
    except Exception as e:
        app.logger.error(f"Failed to send email: {str(e)}")
        return f"Failed to send email: {str(e)}", 500

client = openai.OpenAI(
    api_key="sk-proj-INbeQZ6pQkEB573mEQrMT3BlbkFJl0cYzjdMQbzb7lGBNYkP",
)

def ask_openai(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        print("OpenAI response:", response)

        content = response.choices[0].message.content
        
        return content
    except Exception as e:
        print("Error while communicating with OpenAI API:", e)
        return None
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    messages = [
        {"role": "system", "content": "You are a specialized assistant with expert knowledge in diagnosing, treating, and managing bone fractures only."},
        {"role": "user", "content": user_input}
    ]
    answer = ask_openai(messages)
    print("Answer:", answer)

    if answer:
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'No valid response from OpenAI'})


if __name__ == '__main__':
    app.run(port=5500, debug=True)