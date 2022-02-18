BGANs
========

## Experiment codes for Paper:

Runhai Feng, Dario Grana, Tapan Mukerji, Klaus Mosegaard, "Applicaiton of Bayesian Generative Adversarial Networks to Geological Facies Modeling", Mathematical Geosciences, 2022

## Informations:
It is suggested to use TensorFlow GPU
You can install tensorflow with the following command
```
pip install tensorflow-gpu==1.13.1
```

then run the code
```
./run_bgan.py --batch_size 10 --data_path 'channel' --num_mcmc 2 --num_gen 2 --num_disc 1 --patch_size_x 80 --patch_size_y 80  --out_dir result --train_iter 6001 --optimizer sgd --lr 0.01
```
