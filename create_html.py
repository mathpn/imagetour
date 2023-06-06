import numpy as np
from PIL import Image
from sklearn import svm

with open("./LKH-3.0.8/imagetour.out", "r") as f:
    lkh = f.readlines()

with open("./embeddings.txt", "r") as emb_f:
    emb = emb_f.readlines()

files = [x.split(" ")[0] for x in emb]
emb = [list(map(float, x.split(" ")[1:])) for x in emb]
emb = np.array(emb)

model = svm.LinearSVC(C=1)
labels = np.zeros(len(emb))
labels[0] = 1
model.fit(emb, labels)

similarities = model.decision_function(emb)
sorted_idx = np.argsort(-similarities)

idx = lkh[6:-2]
with open("./tsp_order.html", "w") as html:
    for i in idx:
        # one-indexed
        html.write(f"<img src='./{files[int(i)-1]}' height=40px />")

with open("./svm_order.html", "w") as html:
    for i in sorted_idx:
        html.write(f"<img src='./{files[i]}' height=40px />")
