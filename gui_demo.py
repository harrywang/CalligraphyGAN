from flask import Flask, escape, request, flash, render_template, url_for
from ai_menu import AIMenu
import os

if not os.path.exists('./static/tmp'):
    os.makedirs('./static/tmp')

app = Flask(__name__)
ai_menu = AIMenu(result_path='./static/tmp', topk=10)
print('go to http://localhost:5000 to see the demo')


@app.route('/', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return render_template('cgan.html')
    elif request.method == 'POST':
        desc = request.form['description']
        urls = ai_menu.generate(desc)
        return render_template('cgan.html', url_1=urls['url1'][2:],
                               url_2=urls['url2'][2:])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
