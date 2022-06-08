import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from flask import Flask, render_template, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from prediction.predict import predict_from_sentence
from utils.amr import penman_to_dot
import base64

app = Flask(__name__)

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)

# with Flask-WTF, each web form is represented by a class
# "NameForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class NameForm(FlaskForm):
    sentence = StringField('Kalimat', validators=[DataRequired()], render_kw={"placeholder": "Example: Aku makan buah di pagi hari ini"})
    submit = SubmitField('Parse')

@app.route('/', methods=['GET', 'POST'])
def index():
    path = "web_app/graph_output/output"
    form = NameForm()
    message = ""
    if form.validate_on_submit():
        sentence = form.sentence.data
        amr_graph = predict_from_sentence(sentence)
        dot = penman_to_dot(amr_graph, path)
        with open("{}.png".format(path), "rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())

        flash(str(amr_graph))
        return render_template('index.html', form=form, message=message, graph=encoded_img.decode("UTF-8"))
        # return redirect( request.url )
    return render_template('index.html', form=form, message=message)

if __name__ == '__main__':
	app.run(debug = True, port=5000)