1. Data Structures
-> Find out if there are suitable Datastructures for more efficient Data access, runtime improvement

2. Matrix Multiplication
-> Improved Computations trough Sparse, LU etc might be achievable (or not)
->  Maybe Sparse Matrix in Brain_Neural_Network.py hidden & input matrix, right now is dense

3. Else, see @interactive_sim_obersvations.txt or @README.md 

4. The @profile_sim.py is a good start to find out where most time is spend, but 
    needs more improvement. (Just a quickly AI-generated script)

5. The movement of the predators and preys is not optimal, this definitly needs to be improved
-> this needs to be tested, because in the end we want to have a kind of natural movement of the animals
-> in case that they have developted neural network brains

6. @Brain_Neural_Network.py the activation function and initial weights have maybe to be optimized/adapted
-> maybe there is a bug in the infinity loop function 
            
    ```python
    for i in range(self.num_outputs, ot_hi):
        if np.all(self.Hidden_Matrix[i] == 0):
            indexListe.remove(i) 
            sampleReihnfolge.append(i)
    ```

!!! Removing and iterating over the same list is not a good idea !!!


# OTHER OBSERVATIONS:

Problems:
- new Neurons spwaning in the air 
    
    -> complete nonsense.
    it leads to an overflow of death neurons not in use. 
    New edges spawn to death neurons leads to rarly triggering the output neurons. 

- The sightfield vecotors dont really work. 
    -> only single neuron at a time is triggered. 
    i should search for a different approach. 
    - either same, but improve that several neurons are triggered (when animal gets closer). 
    - or new approach: Maybe with some Angular calculation. 

- For the output angle i have to check, wheter also minus values are possible. (probably -180 to 180 would be reasonable)
    -> I want to be able to turn left and right.

The simulation itself works.
