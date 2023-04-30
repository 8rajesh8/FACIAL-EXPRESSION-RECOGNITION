from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def first():
    return "<h1>Hello Meeeeeeee</h1>"


if (__name__) == ('__main__'):
    app.run()
