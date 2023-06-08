# imagetour

This is heavily inspired by [wordtour](https://github.com/joisino/wordtour), which proposes one-dimensional word embeddings. These are achieve by treating the n-dimensional embedding space as a travelling-salesman problem.

Here, I apply the same concept (and much of the same code, go check the original repo) to CelebA images using a vision transformer model to extract high-dimensional embeddings.

# Sample images

![sample images](https://github.com/mathpn/imagetour/raw/main/assets/example.png?raw=true)

This image shows segments from 1D embeddings obtained using the travelling salesman approach or a single-component PCA. It can be quite subjective, but the TSP approach does seem to produce somewhat better segments. Both 1D embeddings could be potentially improved finetuning the network to our data.

# How to run it

## Download the images

1. Download the CelebA images from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Go into Align&Cropped Images, then Img. Download the img_align_celeba.zip file and unzip it to ./images/celeba
2. Download [LKH](http://webhotel4.ruc.dk/~keld/research/LKH-3/) using the provided script:

```bash
bash download.sh
```

3. Extract the image embeddings (this can be slow without a GPU):

```bash
python extract_emb.py
```

3. Compile the generator (this comes directly from wordtour)

```bash
make
```

4. Solve the travelling-salesman problem: 

```bash
bash solve_tsp.sh ./embeddings.txt
```

5. Visualize the embedding space

You can visualize the 1D embedding space using an HTML file or a PNG image with some samples.

- Generating the HTML file:

```bash
python visualize.py --html
```

- Generating a PNG with sample images (also compares with 1D-embeddings using PCA)

```bash
python visualize.py
```
