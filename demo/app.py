from flask import Flask, render_template, request, get_template_attribute
import numpy as np
import io
import os
import h5py
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import netvlad_ghost_norm
from base64 import b64encode
from sklearn.neighbors import NearestNeighbors
import glob

#L2 Norm layer for after cnn base
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
    
#input transform for the vgg16
def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((480,640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/inference_sync", methods=['POST'])
def inference_sync():
    if request.method == 'POST' and 'formFile' in request.files:
        photo = request.files['formFile']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        img = Image.open(in_memory_file)
        img_transform = transform(img)
        img_transform = img_transform.unsqueeze(0)
        query = model.features(img_transform)
        query, _ = model.pool(query)
        #We want the best 3 matching in the database
        k = 3
        D, I = knn.kneighbors(query.detach().numpy(), k)
        image = []
        for i in range(k):
            image.append({"name": os.path.basename(file_name[I[0][i]]), "index" : i+1, "distance" : D[0][i]})
        prev_image = b64encode(in_memory_file.getvalue()).decode('utf-8')
        sim_images = get_template_attribute('_similar_images.html', 'similar_images')
        html = sim_images(prev_image, image)
        
        return html
    return "<h3>No results</h3>"

def encode_databse(file_name):
    print("Offline encoding for the images in database")
    encode_db = h5py.File("./resources/encode_db.h5", "w")
    db = torch.zeros(len(file_name), 32768)
    
    for i in range(len(file_name)):
        img = Image.open(file_name[i])
        img_transform = transform(img)
        img_transform = img_transform.unsqueeze(0)
        query = model.features(img_transform)
        query, _ = model.pool(query)
        db[i:i+1, :] = query.detach()
    encode_db.create_dataset("database", data = db)
    encode_db.close()
        
        

port = int(os.getenv("PORT", 5000))
if __name__ == '__main__':
    encoder = models.vgg16(pretrained=True)
    layers = list(encoder.features.children())[:-2]

    for l in layers:
        for p in l.parameters():
            p.requires_grad = False
        
    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module("features", encoder)

    netvlad_layer = netvlad_ghost_norm.NetVlad_Ghost(num_clusters=64, dim=512, vladv2=False, normalize_input=True, num_ghost_clusters=6)
    model.add_module("pool", netvlad_layer)
    model.features = nn.DataParallel(model.features)
    model.pool = nn.DataParallel(model.pool)
    transform = input_transform()
    checkpoints = torch.load("resources/netvlad_ghost.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoints["state_dict"])
    file_name = glob.glob("./static/database/*.jpeg")
    '''with open("resources/file_name.txt", "r") as f:
        file_name = f.read()
        file_name = file_name.split("\n")
        f.close()'''
    if not os.path.exists("./resources/encode_db.h5"):
        encode_databse(file_name)
    
    database = h5py.File("resources/encode_db.h5","r")['database']
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database)
    print("Preloading complete")
    app.run(debug = True, host='0.0.0.0', port=port)