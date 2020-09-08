import joblib
import torch
import base64
import config
import matplotlib
import numpy as np
from PIL import Image
from io import BytesIO
from train import MnistModel
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
matplotlib.use('Agg')


MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

memory = joblib.Memory(location="../checkpoint/", verbose=0)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


@memory.cache
def mnist_prediction(img):
    img = img.to(DEVICE, dtype=torch.float)
    outputs = MODEL(x=img)
    probs = torch.exp(outputs.data)[0] * 100
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    ax.set_ylim(0, 110)
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax)
    probimg = BytesIO()
    fig.savefig(probimg, format='png')
    encoded = base64.b64encode(probimg.getvalue()).decode('utf-8')
    _, output = torch.max(outputs.data, 1)
    pred = output.cpu().detach().numpy()
    return pred[0], encoded


@app.route("/process", methods=["GET", "POST"])
def process():
    data_url = str(request.get_data())
    offset = data_url.index(',')+1
    img_bytes = base64.b64decode(data_url[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('L')
    img = img.resize((28, 28))
    # img.save(r'templates\image.png')
    img = np.array(img)
    img = img.reshape((1, 28, 28))
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0)
    data, encoded = mnist_prediction(img)
    response = {
        'data': str(data),
        'encoded': str(encoded),
    }
    return jsonify(response)


@app.route("/", methods=["GET", "POST"])
def start():
    return render_template("default.html")


if __name__ == "__main__":
    MODEL = MnistModel(classes=10)
    MODEL.load_state_dict(torch.load('checkpoint/mnist.pt'))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)
