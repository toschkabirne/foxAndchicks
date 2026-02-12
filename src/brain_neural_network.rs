// ============================================================================
// NEURAL NETWORK BRAIN FOR PREDATOR-PREY SIMULATION
// ============================================================================
// This file implements a growing neural network that evolves over time through
// mutation and selection. Unlike fixed-topology networks, this network can:
// - Add new hidden neurons during mutation
// - Add new connections between neurons
// - Modify existing connection weights
// ============================================================================

use crate::settings::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================
// These functions transform neuron activations, introducing non-linearity
// which is essential for neural networks to learn complex behaviors.
// ============================================================================
const WEIGHT: f32 = 1.0;
const FULLY_CONNECTED: bool = false;
/// Sigmoid activation with adjustable steepness.
///
/// The Sigmoid function maps values to the (0, 1) range.
pub fn sigmoid(x: f32) -> f32 {
    let a = 3.5; // steepness of curve
    let b = 3.0; // offset of curve
    1.0 / (1.0 + (-x * a + b).exp())
}

/// ReLU (Rectified Linear Unit) activation.
pub fn re_ac(x: f32) -> f32 {
    x.max(0.0)
}

/// Hyperbolic tangent activation (general purpose).
///
/// The Hyperbolic tangent function maps values to the (-1, 1) range.
pub fn act_func(x: f32) -> f32 {
    let a = 1.5;
    act_tanh(a * x)
}

pub fn act_tanh(x: f32) -> f32 {
    x.tanh()
}

/// Specialized activation for speed output.
///
/// Specialized activation for speed output, mapping to [0, 1].
pub fn act_speed(x: f32) -> f32 {
    sigmoid(x)
}

/// Specialized activation for angle/turning output.
///
/// Turning symmetric, tanh's range of (-1, 1) (can turn left or right equally).
pub fn act_angle(x: f32) -> f32 {
    x.tanh()
}

// ============================================================================
// NEURAL NETWORK STRUCTURE
// ============================================================================

/// A growing, evolvable neural network with dynamic topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Number of input nodes (sensory inputs from environment)
    pub num_inputs: usize,

    /// Number of output nodes (always 2: speed and turning)
    pub num_outputs: usize,

    /// Bias value multiplied by energy_factor during forward pass.
    pub bias: f32,

    /// Total number of nodes: inputs + bias + outputs + hidden nodes.
    pub neuron_number: usize,

    /// Topological ordering of hidden nodes for evaluation.
    /// Updated during cycle detection when new connections are added.
    pub eval_order: Vec<usize>,

    /// Connection weights from inputs (+ bias) to outputs/hidden nodes.
    ///
    /// Shape: [outputs + hidden count] x [inputs + BIAS]
    /// Row i represents connections TO node (num_inputs + 1 + i)
    /// Column j represents connection FROM input/bias node j
    pub input_matrix: Vec<Vec<f32>>,

    /// Connection weights between output/hidden nodes.
    ///
    /// Shape: [outputs + hidden count] x [outputs + hidden count]
    /// Row i represents connections TO output/hidden node i
    /// Column j represents connection FROM output/hidden node j
    pub hidden_matrix: Vec<Vec<f32>>,

    /// Stores the most recent input vector (for debugging/visualization).
    pub last_inputs: Vec<f32>,

    /// Stores the most recent activation vector (for debugging/visualization).
    pub last_activations: Vec<f32>,
}

impl NeuralNetwork {
    /// Creates a new neural network with minimal topology.
    ///
    /// Initial structure: just inputs, bias, and outputs (no hidden neurons).
    /// Hidden neurons are added later through mutation.
    pub fn new<R: Rng>(
        num_inputs: usize,
        num_outputs: usize,
        mutate: usize,
        bias: f32,
        rng: &mut R,
    ) -> Self {
        let neuron_number = num_inputs + 1 + num_outputs;

        // Create minimal network structure
        let mut nn = Self {
            num_inputs,
            num_outputs,
            bias,
            neuron_number,
            eval_order: Vec::new(),

            // Input matrix: rows for outputs only (no hidden neurons yet)
            // Each row has columns for all inputs + bias
            input_matrix: vec![vec![0.0; num_inputs + 1]; num_outputs],

            // Hidden matrix: square matrix for outputs (no hidden neurons yet)
            // Initialized to zero (no connections)
            hidden_matrix: vec![Vec::new(); num_outputs],

            // Debug/visualization fields
            last_inputs: Vec::new(),
            last_activations: Vec::new(),
        };

        if FULLY_CONNECTED {
            // Initialize the network with full connectivity between input and output layers.
            for row in nn.input_matrix.iter_mut() {
                for val in row.iter_mut() {
                    let mut w = rng.gen_range(-WEIGHT..WEIGHT);
                    if w == 0.0 {
                        w = 0.01;
                    }
                    *val = w;
                }
            }

            for row in nn.hidden_matrix.iter_mut() {
                for val in row.iter_mut() {
                    let mut w = rng.gen_range(-WEIGHT..WEIGHT);
                    if w == 0.0 {
                        w = 0.01;
                    }
                    *val = w;
                }
            }
        }

        for _ in 0..mutate {
            nn.mutate(rng);
        }
        nn
    }

    /// Handles Mutation Logic of the Neural Network
    pub fn mutate<R: Rng>(&mut self, rng: &mut R) {
        // Chance to add a new neuron on an existing connection
        if rng.gen::<f32>() < add_neuron() {
            self.create_connected_neuron(rng);
        }

        // Chance to add a new weight
        if rng.gen::<f32>() < add_weight() {
            self.create_new_connection(rng);
        }

        // Chance to modify an existing weight
        if rng.gen::<f32>() < change_weight() {
            self.change_weight(rng);
        }
    }

    /// Handles Forward Pass of the Neural Network
    pub fn forward_vectorized(&mut self, inputs: &[f32], energy_factor: f32) -> [f32; 2] {
        // 1. Prepare input activations by combining inputs with the bias.
        let mut in_act = Vec::with_capacity(self.num_inputs + 1);
        in_act.extend_from_slice(inputs);
        in_act.push(self.bias * energy_factor);

        // 2. Initial activations for (outputs + hidden) from input_matrix
        let total_neurons_idx = self.neuron_number - (self.num_inputs + 1); // = outputs + hidden count
        let mut activations = vec![0.0; total_neurons_idx];

        for (i, row) in self.input_matrix.iter().enumerate() {
            activations[i] = Self::row_dot(row, &in_act);
        }

        // 3. Hidden evaluation in topological order
        let hidden_act_start = self.num_outputs;
        for &order in &self.eval_order {
            // order is index in activations
            // hidden_matrix columns are hidden-only sources
            let dot = Self::row_dot(&self.hidden_matrix[order], &activations[hidden_act_start..]);
            activations[order] = act_func(activations[order] + dot);
        }

        // 4. Store for debug
        self.last_inputs = in_act;
        self.last_activations = activations.clone();

        // 5. Compute final outputs (indices 0 and 1)
        // output 0 - speed delta
        let out0_dot = Self::row_dot(&self.hidden_matrix[0], &activations[self.num_outputs..]);
        let out0 = act_speed(activations[0] + out0_dot);

        // output 1 - turn delta
        let out1_dot = Self::row_dot(&self.hidden_matrix[1], &activations[self.num_outputs..]);
        let out1 = act_angle(activations[1] + out1_dot);

        [out0, out1]
    }

    /// Debug version of forward pass that returns (outputs, pre_activations)
    /// pre_activations are the raw values before the final activation function.
    pub fn forward_debug(&mut self, inputs: &[f32], energy_factor: f32) -> ([f32; 2], [f32; 2]) {
        // 1. Prepare input activations
        let mut in_act = Vec::with_capacity(self.num_inputs + 1);
        in_act.extend_from_slice(inputs);
        in_act.push(self.bias * energy_factor);

        // 2. Initial activations for (outputs + hidden) from input_matrix
        let total_neurons_idx = self.neuron_number - (self.num_inputs + 1);
        let mut activations = vec![0.0; total_neurons_idx];

        for (i, row) in self.input_matrix.iter().enumerate() {
            activations[i] = Self::row_dot(row, &in_act);
        }

        // 3. Hidden evaluation in topological order
        let hidden_act_start = self.num_outputs;
        for &order in &self.eval_order {
            let dot = Self::row_dot(&self.hidden_matrix[order], &activations[hidden_act_start..]);
            activations[order] = act_func(activations[order] + dot);
        }

        // 4. Store for debug
        self.last_inputs = in_act;
        self.last_activations = activations.clone();

        // 5. Compute final outputs (indices 0 and 1)
        // output 0 - speed delta
        let out0_dot = Self::row_dot(&self.hidden_matrix[0], &activations[self.num_outputs..]);
        let pre_speed = activations[0] + out0_dot;
        let out0 = act_speed(pre_speed);

        // output 1 - turn delta
        let out1_dot = Self::row_dot(&self.hidden_matrix[1], &activations[self.num_outputs..]);
        let pre_angle = activations[1] + out1_dot;
        let out1 = act_angle(pre_angle);

        ([out0, out1], [pre_speed, pre_angle])
    }

    /// Helper function: computes dot product of a weight row with an activation vector.
    fn row_dot(row: &[f32], vec: &[f32]) -> f32 {
        row.iter().zip(vec.iter()).map(|(w, v)| w * v).sum()
    }

    /// Adds a new hidden unconnected neuron to the network.
    pub fn add_neuron(&mut self) {
        // Add new row to input matrix (connections from inputs/bias to new neuron)
        self.input_matrix.push(vec![0.0_f32; self.num_inputs + 1]);

        // Add new column to all existing rows in hidden matrix
        // (allows connections from new neuron to existing hidden/output nodes)
        for row in &mut self.hidden_matrix {
            row.push(0.0_f32);
        }

        // Add new row to hidden matrix (connections TO new neuron)
        // Width = hidden_count AFTER the column push above.
        let new_width = self.hidden_matrix[0].len();
        self.hidden_matrix.push(vec![0.0_f32; new_width]);

        // Update total neuron count
        self.neuron_number += 1;
    }

    /// Adds a new connection to the network.
    pub fn add_connection<R: Rng>(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: Option<f32>,
        rng: &mut R,
    ) {
        // Use provided weight or generate small random weight
        let mut w = weight.unwrap_or_else(|| rng.gen_range(-WEIGHT..WEIGHT));

        if w == 0.0 {
            w = 0.01;
        }

        let in_bi = self.num_inputs + 1; // First output/hidden node ID
        let hidden_start = in_bi + self.num_outputs; // First hidden node ID

        // Determine which matrix to update based on source type
        if source_id < in_bi {
            // Source is input/bias -> update input_matrix
            if let Some(row) = self.input_matrix.get_mut(target_id - in_bi) {
                if let Some(val) = row.get_mut(source_id) {
                    *val = w;
                }
            }
        } else {
            // Source is output/hidden -> hidden_matrix stores ONLY hidden sources.
            // Outputs are not representable here, so ignore output-as-source.
            if source_id < hidden_start {
                return;
            }
            if let Some(row) = self.hidden_matrix.get_mut(target_id - in_bi) {
                let col = source_id - hidden_start;
                if let Some(val) = row.get_mut(col) {
                    *val = w;
                }
            }
        }
    }

    ///  ----------------------------------------------------
    /// HELPER FUNCTIONS - Mutation
    ///  ----------------------------------------------------
    /// Split an existing connection by creating a new neuron in between.
    fn create_connected_neuron<R: Rng>(&mut self, rng: &mut R) {
        let in_bi = self.num_inputs + 1;
        let hidden_start = in_bi + self.num_outputs;

        let mut connections = Vec::new();

        // Collect all existing connections
        // From input matrix
        for (r, row) in self.input_matrix.iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                if w != 0.0 {
                    connections.push((c, in_bi + r, w));
                }
            }
        }
        // From hidden matrix
        for (r, row) in self.hidden_matrix.iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                if w != 0.0 {
                    // c indexes hidden sources only
                    connections.push((hidden_start + c, in_bi + r, w));
                }
            }
        }

        // Select a random connection to split: source -> target becomes source -> new -> target.
        if !connections.is_empty() {
            let idx = rng.gen_range(0..connections.len());
            let (source, target, weight) = connections[idx];

            if source < in_bi {
                self.input_matrix[target - in_bi][source] = 0.0;
            } else {
                // source is guaranteed hidden here (comes from connections list)
                let col = source - hidden_start;
                self.hidden_matrix[target - in_bi][col] = 0.0;
            }

            self.add_neuron();
            let new_neuron_id = self.neuron_number - 1;

            self.add_connection(source, new_neuron_id, Some(weight), rng);
            self.add_connection(new_neuron_id, target, Some(weight), rng);
        }
    }

    /// Creates a new connection between two existing neurons, preventing cycles.
    fn create_new_connection<R: Rng>(&mut self, rng: &mut R) {
        let in_bi = self.num_inputs + 1;
        let hidden_start = in_bi + self.num_outputs;

        // source can be any input or hidden node (not output)
        // range: [0, in_bi) U [hidden_start, total)
        let mut valid_sources: Vec<usize> = (0..in_bi).collect();
        valid_sources.extend(hidden_start..self.neuron_number);

        let source = valid_sources[rng.gen_range(0..valid_sources.len())];
        // Target can be any hidden or output node
        let target = rng.gen_range(in_bi..self.neuron_number);

        // Check if connection exists
        let connected = if source < in_bi {
            self.input_matrix[target - in_bi][source] != 0.0
        } else {
            // source is hidden => map to hidden_matrix column
            self.hidden_matrix[target - in_bi][source - hidden_start] != 0.0
        };

        if !connected {
            // Check cycles
            if !self.kahn_algorithm(source, target) {
                self.add_connection(source, target, None, rng);
            }
        }
    }

    /// Changes random weights
    fn change_weight<R: Rng>(&mut self, rng: &mut R) {
        let in_bi = self.num_inputs + 1;

        let hidden_start = in_bi + self.num_outputs;
        let mut connections: Vec<(usize, usize)> = Vec::new();

        // Scan input_matrix: input/bias -> output/hidden
        for (r, row) in self.input_matrix.iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                if w != 0.0 {
                    let source = c;
                    let target = in_bi + r;
                    connections.push((source, target));
                }
            }
        }

        // Scan hidden_matrix: hidden -> output/hidden
        for (r, row) in self.hidden_matrix.iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                if w != 0.0 {
                    let source = hidden_start + c;
                    let target = in_bi + r;
                    connections.push((source, target));
                }
            }
        }
        if connections.is_empty() {
            return;
        }

        let (source, target) = connections[rng.gen_range(0..connections.len())];
        let w_ref = if source < in_bi {
            &mut self.input_matrix[target - in_bi][source]
        } else {
            &mut self.hidden_matrix[target - in_bi][source - hidden_start]
        };
        *w_ref += rng.gen_range(-MUT_CHANGE_STEP..MUT_CHANGE_STEP);

        // Enforce non-zero weight
        if *w_ref == 0.0 {
            *w_ref = 0.01;
        }
    }

    // /// HELPER FUNCTIONS - CHECKS FOR CYLES
    // ///  ----------------------------------------------------

    // Cycle detection logic
    pub fn infinity_loop(&mut self, source: usize, target: usize) -> bool {
        let in_bi = self.num_inputs + 1;
        let in_bi_out = in_bi + self.num_outputs;
        let hidden_start = in_bi_out;

        // Check for cycles only if both nodes are in the hidden layer
        if source >= in_bi_out && target >= in_bi_out {
            let ot_hi = self.neuron_number - in_bi; // outputs + hidden count

            // Temporarily add the connection to check for cycles
            self.hidden_matrix[target - in_bi][source - hidden_start] = 1.0;

            let mut sample_order: Vec<usize> = Vec::new();
            let mut index_list: Vec<usize> = (self.num_outputs..ot_hi).collect(); // hidden indices only

            // Remove nodes with no incoming edges from the hidden layer
            let mut k = 0;
            while k < index_list.len() {
                let i = index_list[k];
                // Check if all incoming weights from hidden layer are 0
                if self.hidden_matrix[i].iter().all(|&v| v == 0.0) {
                    index_list.remove(k);
                    sample_order.push(i);
                } else {
                    k += 1;
                }
            }

            while !index_list.is_empty() {
                let before = index_list.len();
                let mut j = 0;

                while j < index_list.len() {
                    let i = index_list[j];
                    // Check if the node has any incoming connections from the remaining set

                    let no_incoming = index_list.iter().all(|&col| {
                        // col is activation index; convert to hidden_matrix column index
                        self.hidden_matrix[i][col - self.num_outputs] == 0.0
                    });

                    if no_incoming {
                        index_list.remove(j);
                        sample_order.push(i);
                    } else {
                        j += 1;
                    }
                }

                if before == index_list.len() {
                    // Cycle detected; revert temporary connection
                    self.hidden_matrix[target - in_bi][source - hidden_start] = 0.0;
                    return true;
                }
            }

            self.eval_order = sample_order;
            false
        } else {
            false
        }
    }

    /// Optimized Kahn's algorithm for cycle detection.
    /// Equivalent to infinity_loop behavior.
    pub fn kahn_algorithm(&mut self, source: usize, target: usize) -> bool {
        let in_bi = self.num_inputs + 1;
        let in_bi_out = in_bi + self.num_outputs;
        let hidden_start = in_bi_out;

        // Check if both are hidden nodes
        if source >= in_bi_out && target >= in_bi_out {
            // Map global indices to 0..hidden_count range
            // Global index `g` -> Local index `g - hidden_start`
            // Row in hidden_matrix: `target - in_bi`
            // Col in hidden_matrix: `source - hidden_start`

            let hidden_count = self.neuron_number - hidden_start;

            // 1. Temporarily add edge (Side effect preserved)
            self.hidden_matrix[target - in_bi][source - hidden_start] = 1.0;

            // 2. Build Adjacency List & In-Degree for HIDDEN sub-graph
            let mut adj = vec![Vec::new(); hidden_count];
            let mut in_degree = vec![0; hidden_count];

            // Iterate only over hidden rows in hidden_matrix
            // Rows: self.num_outputs .. (self.num_outputs + hidden_count)
            for (r_local, degree) in in_degree.iter_mut().enumerate().take(hidden_count) {
                let r_matrix = self.num_outputs + r_local;
                // r_matrix is index in hidden_matrix
                // Corresponds to node `hidden_start + r_local`

                for (c_local, &w) in self.hidden_matrix[r_matrix].iter().enumerate() {
                    if w != 0.0 {
                        // Edge from c_local to r_local
                        adj[c_local].push(r_local);
                        *degree += 1;
                    }
                }
            }

            // 3. Kahn's Algorithm
            let mut queue = Vec::new();
            for (i, &degree) in in_degree.iter().enumerate().take(hidden_count) {
                if degree == 0 {
                    queue.push(i);
                }
            }

            let mut sorted_order = Vec::with_capacity(hidden_count);

            while let Some(u) = queue.pop() {
                sorted_order.push(u);

                for &v in &adj[u] {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push(v);
                    }
                }
            }

            if sorted_order.len() < hidden_count {
                // Cycle detected
                // Revert dummy edge
                self.hidden_matrix[target - in_bi][source - hidden_start] = 0.0;
                true
            } else {
                // No cycle. Update eval_order.
                // eval_order expects indices in 'activations' list (relative to output start?)
                // infinity_loop pushes `i` from `index_list`.
                // `index_list` initialized with `(self.num_outputs..ot_hi)`
                // So `sorted_order` (0..hidden_count) needs to be mapped back to `self.num_outputs..`
                self.eval_order = sorted_order.iter().map(|&x| x + self.num_outputs).collect();
                false
            }
        } else {
            false
        }
    }

    // ====================================================================
    // COMPLEXITY METRICS (for NEAT analysis)
    // ====================================================================

    /// Returns the total number of neurons in the network
    /// (inputs + bias + outputs + hidden).
    pub fn num_neurons(&self) -> usize {
        self.neuron_number
    }

    /// Returns the number of non-zero connections (weights) in the network.
    pub fn num_connections(&self) -> usize {
        let input_conns: usize = self
            .input_matrix
            .iter()
            .map(|row| row.iter().filter(|&&w| w != 0.0).count())
            .sum();
        let hidden_conns: usize = self
            .hidden_matrix
            .iter()
            .map(|row| row.iter().filter(|&&w| w != 0.0).count())
            .sum();
        input_conns + hidden_conns
    }
}
