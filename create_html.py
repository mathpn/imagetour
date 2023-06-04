import numpy as np
from PIL import Image

with open("./LKH-3.0.8/imagetour.out", "r") as f:
    lkh = f.readlines()

idx = lkh[6:-2]
with open("./order.html", "w") as html:
    for i in idx:
        html.write(f"<img src='./images/img_{int(i) - 1}.jpg' height=40px />")
