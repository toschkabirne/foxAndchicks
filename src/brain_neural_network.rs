// ============================================================================
// NEURAL NETWORK BRAIN FOR PREDATOR-PREY SIMULATION
// ============================================================================
// This file implements a growing neural network that evolves over time through
// mutation and selection. Unlike fixed-topology networks, this network can:
// - Add new hidden neurons during mutation
// - Add new connections between neurons
// - Modify existing connection weights
// ============================================================================

use crate::settings;
use rand::Rng;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================
// These functions transform neuron activations, introducing non-linearity
// which is essential for neural networks to learn complex behaviors.
// ============================================================================

/// Sigmoid activation with adjustable steepness.
///
/// Design rationale: Sigmoid squashes values to (0, 1) range. The parameters
/// (a = 1.5, b = 0.0) control steepness and offset.
pub fn sigmoid(x: f32) -> f32 {
    let a = 1.5;
    let b = 0.0;
    1.0 / (1.0 + (-x * a - b).exp())
}

/// ReLU (Rectified Linear Unit) activation.
pub fn re_ac(x: f32) -> f32 {
    x.max(0.0)
}

/// Hyperbolic tangent activation (general purpose).
///
/// Design rationale: tanh squashes to (-1, 1) range
pub fn act_func(x: f32) -> f32 {
    x.tanh()
}

/// Specialized activation for speed output.
///
/// Design rationale: Speed must be in [0, 1] range (can't move backward).
/// - Input 0 produces sigmoid(-1.5) ≈ 0.18 (slow movement)
/// - Input needs to be > 1.5 to get significant speed
pub fn act_speed(x: f32) -> f32 {
    sigmoid(x - 1.5)
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
#[derive(Clone, Debug)]
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
    /// Updated by infinity_loop() when new connections are added.
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
            hidden_matrix: vec![vec![0.0; num_outputs]; num_outputs],

            // Debug/visualization fields
            last_inputs: Vec::new(),
            last_activations: Vec::new(),
        };

        // Apply initial mutations to create diversity
        for _ in 0..mutate {
            nn.mutate(rng);
        }
        nn
    }

    /// Adds a new hidden neuron to the network.
    ///
    /// This grows both matrices to accommodate the new neuron:
    /// - Adds a row to input_matrix (new neuron can receive from inputs)
    /// - Adds a row and column to hidden_matrix (new neuron can connect to/from
    ///   other hidden/output nodes)
    ///
    /// Design rationale: New neurons start with all-zero connections. Subsequent
    /// mutations will add actual connections with non-zero weights. This prevents
    /// the new neuron from immediately affecting network behavior, allowing
    /// gradual integration.
    ///
    /// The new neuron is added during "add neuron" mutation, which typically
    /// splits an existing connection: A -> B becomes A -> new -> B.
    pub fn add_neuron(&mut self) {
        // Add new row to input matrix (connections from inputs/bias to new neuron)
        self.input_matrix.push(vec![0.0_f32; self.num_inputs + 1]);

        // Add new column to all existing rows in hidden matrix
        // (connections from new neuron to existing hidden/output nodes)
        for row in &mut self.hidden_matrix {
            row.push(0.0_f32);
        }

        // Add new row to hidden matrix (connections to new neuron)
        let new_width = self.hidden_matrix.len() + 1;
        self.hidden_matrix.push(vec![0.0_f32; new_width]);

        // Update total neuron count
        self.neuron_number += 1;
    }

    /// Adds or modifies a connection weight between two neurons.
    ///
    /// Design rationale for node indexing:
    /// The function uses the global node ID scheme:
    /// - source_id < in_bi: Source is an input or bias node
    /// - source_id >= in_bi: Source is an output or hidden node
    /// - target_id is always >= in_bi (outputs/hidden only)
    ///
    /// When modifying matrices, we convert global IDs to matrix indices:
    /// - For input_matrix[target - in_bi][source]: target row, source column
    /// - For hidden_matrix[target - in_bi][source - in_bi]: both adjusted
    ///
    /// # Arguments
    /// * `source_id` - Global ID of source neuron
    /// * `target_id` - Global ID of target neuron  
    /// * `weight` - Optional weight value (if None, generates random weight in [-0.2, 0.2])
    /// * `rng` - Random number generator
    pub fn add_connection<R: Rng>(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: Option<f32>,
        rng: &mut R,
    ) {
        // Use provided weight or generate small random weight
        let w = weight.unwrap_or_else(|| rng.gen_range(-0.2..0.2));

        let in_bi = self.num_inputs + 1; // First output/hidden node ID

        // Determine which matrix to update based on source type
        if source_id < in_bi {
            // Source is input/bias -> update input_matrix
            if let Some(row) = self.input_matrix.get_mut(target_id - in_bi) {
                if let Some(val) = row.get_mut(source_id) {
                    *val = w;
                }
            }
        } else {
            // Source is output/hidden -> update hidden_matrix
            if let Some(row) = self.hidden_matrix.get_mut(target_id - in_bi) {
                if let Some(val) = row.get_mut(source_id - in_bi) {
                    *val = w;
                }
            }
        }
    }

    pub fn mutate<R: Rng>(&mut self, rng: &mut R) {
        let in_bi = self.num_inputs + 1;

        // Add Neuron
        if rng.gen::<f32>() < settings::add_neuron() {
            let mut connections = Vec::new();

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
                        connections.push((in_bi + c, in_bi + r, w));
                    }
                }
            }

            if !connections.is_empty() {
                let idx = rng.gen_range(0..connections.len());
                let (source, target, weight) = connections[idx];

                // Disable old connection
                if source < in_bi {
                    self.input_matrix[target - in_bi][source] = 0.0;
                } else {
                    self.hidden_matrix[target - in_bi][source - in_bi] = 0.0;
                }

                self.add_neuron();
                let new_neuron_id = self.neuron_number - 1;

                // Add Source -> New -> Target
                self.add_connection(source, new_neuron_id, Some(weight), rng);
                self.add_connection(new_neuron_id, target, Some(weight), rng);
            }
        }

        // Add Weight
        if rng.gen::<f32>() < settings::add_weight() {
            let hidden_start = in_bi + self.num_outputs;

            // source can be any input or hidden node (not output)
            // range: [0, in_bi) U [hidden_start, total)
            let mut valid_sources: Vec<usize> = (0..in_bi).collect();
            valid_sources.extend(hidden_start..self.neuron_number);

            let source = valid_sources[rng.gen_range(0..valid_sources.len())];
            // target can be any hidden or output node
            // range: [in_bi, total)
            let target = rng.gen_range(in_bi..self.neuron_number);

            // Check if connection exists
            let connected = if source < in_bi {
                self.input_matrix[target - in_bi][source] != 0.0
            } else {
                self.hidden_matrix[target - in_bi][source - in_bi] != 0.0
            };

            if !connected {
                // Check cycles
                if !self.infinity_loop(source, target) {
                    self.add_connection(source, target, None, rng);
                }
            }
        }

        // Change Weight
        if rng.gen::<f32>() < settings::change_weight() {
            for _ in 0..4 {
                let hidden_start = in_bi + self.num_outputs;
                let mut valid_sources: Vec<usize> = (0..in_bi).collect();
                valid_sources.extend(hidden_start..self.neuron_number);

                let source = valid_sources[rng.gen_range(0..valid_sources.len())];
                let target = rng.gen_range(in_bi..self.neuron_number);

                let w_ref = if source < in_bi {
                    &mut self.input_matrix[target - in_bi][source]
                } else {
                    &mut self.hidden_matrix[target - in_bi][source - in_bi]
                };

                if *w_ref != 0.0 {
                    *w_ref += rng.gen_range(-0.05..0.05);
                    break;
                }
            }
        }
    }

    /// Helper function: computes dot product of a weight row with an activation vector.
    ///
    /// Design rationale: This is the core operation in neural network forward pass:
    /// output = sum(weight[i] * activation[i] for all i)
    ///
    /// Using a separate function:
    /// 1. Improves readability (forward_vectorized is complex enough)
    /// 2. Could be optimized separately (e.g., SIMD) in the future
    /// 3. Clearly shows this is standard linear algebra
    fn row_dot(row: &[f32], vec: &[f32]) -> f32 {
        row.iter().zip(vec.iter()).map(|(w, v)| w * v).sum()
    }

    pub fn forward_vectorized(&mut self, inputs: &[f32], energy_factor: f32) -> [f32; 2] {
        // 1. Prepare input activations
        // We can reuse a pre-allocated vector if we want, but let's just make a new one for safety first
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
        // eval_order contains indices into 'activations' list (which corresponds to hidden/output nodes)
        for &order in &self.eval_order {
            // order is index in activations
            let dot = Self::row_dot(&self.hidden_matrix[order], &activations);
            activations[order] = act_func(activations[order] + dot);
        }

        // 4. Store for debug
        self.last_inputs = in_act;
        self.last_activations = activations.clone();

        // 5. Compute final outputs (indices 0 and 1)
        // output 0 - speed delta
        let out0_dot = Self::row_dot(&self.hidden_matrix[0], &activations);
        let out0 = act_speed(activations[0] + out0_dot);

        // output 1 - turn delta
        let out1_dot = Self::row_dot(&self.hidden_matrix[1], &activations);
        let out1 = act_angle(activations[1] + out1_dot);

        [out0, out1]
    }

    // Preserving original algorithm logic for cycle detection as requested
    pub fn infinity_loop(&mut self, source: usize, target: usize) -> bool {
        let in_bi = self.num_inputs + 1;
        let in_bi_out = in_bi + self.num_outputs;

        // only if BOTH are hidden ids (because inputs->hidden or hidden->output can't form simple cycles in this feedforward-ish structure unless back connections exist)
        // logic from Python: check if adding edge source->target creates cycle
        if source >= in_bi_out && target >= in_bi_out {
            let ot_hi = self.neuron_number - in_bi; // outputs + hidden count

            // dummy insert
            self.hidden_matrix[target - in_bi][source - in_bi] = 1.0;

            let mut sample_order: Vec<usize> = Vec::new();
            let mut index_list: Vec<usize> = (self.num_outputs..ot_hi).collect(); // hidden indices only

            // remove "null rows" (nodes with no incoming edges from hidden layer)
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
                    // check incoming from remaining hidden nodes
                    // ie. is there any node in 'index_list' that has a connection to 'i'?
                    // The logic here seems to checking if `i` has NO incoming connections from the *remaining* pool.

                    let no_incoming = index_list
                        .iter()
                        .all(|&col| self.hidden_matrix[i][col] == 0.0);

                    if no_incoming {
                        index_list.remove(j);
                        sample_order.push(i);
                    } else {
                        j += 1;
                    }
                }

                if before == index_list.len() {
                    // cycle detected -> revert dummy
                    self.hidden_matrix[target - in_bi][source - in_bi] = 0.0;
                    return true;
                }
            }

            self.eval_order = sample_order;
            false
        } else {
            false
        }
    }
}
