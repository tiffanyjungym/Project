from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    name = TextField('Product page link:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def my_form():
    form = ReusableForm(request.form)

    print form.errors
    if request.method == 'POST':
        name = request.form['name']
        print name

        if form.validate():
            # Save the comment here.
            flash('Hello ' + name)
        else:
            flash('All the form fields are required. ')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()