from flask import Flask, request, redirect, url_for, flash, render_template
import os
from werkzeug.utils import secure_filename
from leafDiseaseDetection import predict_disease

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file was submitted')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file was selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            prediction = predict_disease(filepath)
            return render_template('result.html', result=prediction)

    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)