# TODOs

Make sure to quickly mark something if you are working on it. Just put your name after the task and check the box after you finished it. :)

## DataManager

- [ ] Also store metadata s.a. max_preds, max_preys, init_NN_weights etc.

## Animal

- [x] Create animal class that predator and prey inherit from
- [x] Is sight possible accross borders? It seems like preys tend to cluster at the borders

## Neural Network

- [ ] Use fully connected NN
- [ ] Initial weights: Create a simple NN that serves as a working starting point for evolution
- [ ] Activation function: Check the behavior of different activation functions (e.g. sigmoid, ReLU, tanh)
- [ ] Check if we can improve matrix multiplication (e.g. sparse matrix, decomposition etc..)
    - [ ] Maybe Sparse Matrix in Brain_Neural_Network.py hidden & input matrix, right now is dense
- [ ] Check if we should improve sightlines
    - [ ] not only one line activated when animal gets close
    - [ ] or new approach: maybe with some Angular calculation

## Game

- [x] Add a slider to playback visualisation
- [ ] Add graphs
- [ ] toroidal wrap for spawns

## General

- [ ] Profiling: Check where the code spends the most time and why
