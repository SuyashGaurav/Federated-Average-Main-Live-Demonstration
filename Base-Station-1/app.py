from flask import Flask, render_template, Response, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random, time, re
import pandas as pd
import requests
import os
import io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['seq_no'] = 0
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(50), nullable = False)
    file_content = db.Column(db.String(50), nullable = False)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        try:
            f = open(f'cached_file/{task_content}.txt', 'r')
            file_content = f.readline()
            status = "File fetched from cache"
            f.close()
        except:
            url = f"http://10.196.11.11:5001/download/{task_content}"
            r = requests.get(url, data={'content': f"{task_content}"}, timeout=60)
            file_content = r.content.decode('utf-8')
            status = "File fetched from central server"
        

        file = open('DPP/Datasets/dataset.txt', 'a')
        file.write(str(int(time.time())*100 + random.randint(0, 500))+" "+str(task_content)+" "+str(random.randint(1000, 10000)) + "\n")
        file.close()
        new_task = Todo(content=task_content, status = status, file_content = file_content)
        db.session.add(new_task)
        db.session.commit()
        return redirect('/')
        
    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        last_50_tasks = tasks[-min(len(tasks),15):]
        return render_template('index.html', tasks=last_50_tasks)
    
@app.route('/images/<filename>')
def serve_image(filename):
    # Specify the path to the directory where your images are stored outside of your Flask project
    image_dir = 'DPP/experiments/431ad8fae2/'
    # Use send_file to serve the image
    return send_file(image_dir + filename, mimetype='image/jpg')

@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return "There is an issue in deleting that task"
    
    
@app.route('/download/<string:content>')
def download(content):
    file_path = f'cached_file/{content}.txt'
    if os.path.isfile(file_path):
        sendfile = send_file(file_path, as_attachment=True)
        return sendfile
    else:
        url = f"http://10.196.11.11:5001/download/{content}"
        r = requests.get(url, data={'content': f"{content}"}, timeout=60)   
        file_data = io.BytesIO(r.content)
        return send_file(file_data, as_attachment=True, download_name=f"{content}.txt")
    

if __name__ == "__main__":
    with app.app_context():
        db.drop_all()
        db.create_all()
    f = open('DPP/Datasets/dataset.txt', 'w')
    f.write("0 0 0\n")
    f.close()
    app.run(host='0.0.0.0', port=5000, debug=True)