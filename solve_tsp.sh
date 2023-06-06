#!/bin/bash

./make_LKH_file $1 $(cat $1 | wc -l) > ./LKH-3.0.8/imagetour.tsp
cd ./LKH-3.0.8
./LKH imagetour.par
cd ..
python create_html.py
