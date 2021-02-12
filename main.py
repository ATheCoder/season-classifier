import json
import urllib.request
from flask import Flask
from flask import request
from fastbook import *
from fastai import *
from fastai.vision.core import *
app = Flask(__name__)
@app.route('/', methods=['POST'])
def handler():
    defaults.device = torch.device('cpu')
    path = Path('./model.pkl')
    learner = load_learner(path)
    file = request.files['file']
    file.save('./tmp/image.jpg')
    
    #image = request.args['image']
    #urllib.request.urlretrieve(image, './tmp/image.jpg')
    imgs = get_image_files('./tmp')
    img = PILImage.create(imgs[0])
    pred_class,pred_idx,outputs = learner.predict(img)
    print(pred_class)
    print(outputs)
    print(learner.dls.vocab)
    return json.dumps({
        "predictions": sorted(
            zip(learner.dls.vocab, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })
