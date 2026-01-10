# Evolutionary Predator-Prey Simulation

An advanced simulation of predator-prey dynamics powered by evolving neural networks. Each entity (Predator and Prey) possesses a unique brain that determines its behavior based on sensory inputs, evolving over generations through mutation and natural selection.

## 🌟 Features

- **Neural Network Brains**: Entities are controlled by dynamic neural networks with evolving architectures (Input -> Hidden -> Output).
- **Genetic Evolution**: Survival results in reproduction where offspring inherit and mutate their parent's neural weights and structures.
- **Ray-Cast Vision**: Prey and predators use multi-directional "sights" to navigate and detect targets or threats.
- **Optimized Performance**: Uses Spatial Hashing for efficient $O(n)$ proximity queries, allowing for hundreds of simultaneous entities.
- **Interactive Mode**: Take control of a predator yourself to test the survival strategies of the evolving prey.
- **Parameter Search Pipeline**: Includes a headless runner and grid-search orchestrator to find the most stable evolutionary parameters.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [NumPy](https://numpy.org/) (for vectorized NN calculations)
- [Pygame](https://www.rsgame.org/) (for visualization)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd PredatorPray2or1

# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install numpy pygame
```

## 🎮 How to Run

### 1. Main Simulation (Visual)
Run the standard simulation with Pygame rendering.
```bash
python main.rs
```

### 2. Interactive  Testing Mode
Control a "Predator" with your arrow keys and observe how the prey moves to avoid you, this is for testing purpose.
```bash
python interactive_sim.rs
```

### 3. Parameter Search (Headless)
Run a grid search to find parameters that maximize simulation stability and duration.
## Here you must set the variable PYTHON_EXECUTABLE = "path/to/virtualenvFolder/venv/bin/python"
```bash
python parameter_search.rs
```

### 4. Profiling the functions
Profiling the efficeny of the functions, Where is most time spend? 
```bash
python profile_sim.rs
```

### 5. Khans Test
This script is only for test purpose, comparing the implemented code logic how signals are passed in a NN with a known logic: Khans cycle algorithm. This is not needed.
```bash
python Khans_test.rs
```

## 🧠 Brain & Evolution

The core of the simulation is the `NeuralNetwork` class in `module/Brain_Neural_Network.rs`.

### Mutation Parameters
The simulation optimizes for three critical mutation rates:
- **ADD_NEURON**: Chance to add a new hidden neuron.
- **ADD_WEIGHT**: Chance to create a new connection between existing neurons.
- **CHANGE_WEIGHT**: Chance to perturb an existing connection's weight.

### Best Found Parameters
So Far best Parameters are (this is by far not the best, but we have to start somewhere):
- `ADD_NEURON`: 0.1
- `ADD_WEIGHT`: 0.5
- `CHANGE_WEIGHT`: 0.9

## 🛠 Project Structure

- `main.rs`: for the main visual simulation.
- `interactive_sim.rs`: to test movement.
- `headless_runner.rs`: Lightweight simulation script for data gathering.
- `parameter_search.rs`: Orchestrator for grid-searching the parameter space.
- `module/`:
  - `Animals.rs`: Core logic for Predator and Prey entities.
  - `Brain_Neural_Network.rs`: Vectorized Neural Network implementation with mutation logic.
  - `settings.rs`: Global simulation constants and starting values.

## 📝 Observations & Problems
Current areas for improvement (tracked in `interactive_sim_obersvations.txt`):
- Improving sight field vectors for multi-neuron triggering.
- Refining output angle calculations for better maneuverability.
- Handling "dead" neurons in evolving architectures.


