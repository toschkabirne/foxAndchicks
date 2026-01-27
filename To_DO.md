# TODOs

Make sure to quickly mark something if you are working on it. Just put your name after the task and check the box after you finished it. :)

## DataManager

- [ ] Also store metadata s.a. max_preds, max_preys, init_NN_weights etc.


## Animal

- [x] Create animal class that predator and prey inherit from
- [x] Is sight possible accross borders? It seems like preys tend to cluster at the borders
- [ ] in move_step the threashold for speed_factor is an essential logic which might need some nice fine tuning

## Neural Network

- [x] Use fully connected NN
- [x] Initial weights: Create a simple NN that serves as a working starting point for evolution
- [ ] Activation function: Check the behavior of different activation functions (e.g. sigmoid, ReLU, tanh)
- [ ] Check if we can improve matrix multiplication (e.g. sparse matrix, decomposition etc..)
    - [ ] Maybe Sparse Matrix in Brain_Neural_Network.py hidden & input matrix, right now is dense
- [ ] We could change the logic of the sightlines, not binary 0/1, but additionally dependent on the distance a value in [0,1]
    - Note: This might be a bigger change, which needs to be carefully considered

## Game

- [x] Add a slider to playback visualisation
- [x] Add graphs
- [x] toroidal wrap for spawns
- [ ] On Click onto a animal shows:
    - [x] its Neural Network
    - [ ] its energy
    - [ ] for predators: digestive cooldown (REPRO_COOLDOWN_FRAMES)
    
    

## General

- [ ] Profiling: Check where the code spends the most time and where
    - [ ] game_heat_allo vs game to compare if removing heap allocations gives speed up // Anton will take care of this
- [ ] Add a lot of asserts (they verify that the code is correct and Prof. Koch likes them) 
- [ ] Check TODOs together

