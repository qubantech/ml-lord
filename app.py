import shutil
from flask import Flask, request, send_file
import torch

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)


@app.route('/', methods=['POST'])
def hello_world():
    shutil.rmtree('./runs/detect')
    data = request.json
    results = model(data['link'])
    results.save()
    name = data['link'].split('/')[-1]
    filename = f'./runs/detect/exp/{name}'
    return send_file(filename, mimetype='image/gif')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
