from model import *
from flask import Flask, render_template, request

model = rf()
print("__________Model trained__________")

app = Flask(__name__)


@app.route("/")
def root():
    return render_template('index.html')


@app.route('/search', methods=['post'])
def search():
    q = request.form['query']
    search_results = model.search(q).head(20)
    docs = merge(search_results)
    return render_template('search.html', query=q, result=docs)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')
