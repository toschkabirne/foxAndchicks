use crate::settings;
use rand::Rng;

// ----- Activation functions (mirrors Python) -----
pub fn sigmoid(x: f32) -> f32 {
    let a = 3.0;
    let b = 0.0;
    1.0 / (1.0 + (-x * a - b).exp())
}

pub fn re_ac(x: f32) -> f32 {
    x.max(0.0)
}

pub fn act_func(x: f32) -> f32 {
    x.tanh()
}

pub fn act_speed(x: f32) -> f32 {
    0.0_f32.max(x.tanh())
}

pub fn act_angle(x: f32) -> f32 {
    x.tanh()
}

// Replaced custom rounding with standard round() for simplicity, unless exact Python compat is strictly required.
// However, the original code had a very specific "ties to even" implementation.
// Standard Rust f32::round() rounds half-way cases away from zero (mostly).
// If "ties to even" is critical, we can keep using a helper but maybe clean it up.
// given the user instruction "optimize possibly bulky ... code", I will simplify this to standard round
// UNLESS it's critical. The comments said "We replicate for the one place it matters", so let's stick to standard round
// but if I wanted to be safe I'd keep it. user said "Remove Python specific helpers like round_ties_even_to_i32 (use standard rounding)".
// So I will remove it.

// ----- Neural Network -----
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub bias: f32,
    pub neuron_number: usize, // total neurons: inputs + bias + outputs + hidden

    // topological order for hidden nodes (indices in activations vector)
    pub eval_order: Vec<usize>,

    // Stored row-major. Width = outputs + hidden (dynamic).
    // Note: since the size of 'hidden' grows, this matrix grows in both dimensions.
    // Ideally we'd valid usage, but for now we mimic the growing behavior.
    // Actually, keeping it as Vec<Vec<f32>> might be easier for resizing, BUT
    // we want optimization.
    // However, resizing a flat matrix 'in-place' when rows/cols are added is tricky/expensive (inserts).
    // Given the mutation logic adds neurons frequently, maybe Vec<Vec<f32>> IS better for the growing phase,
    // or we pre-allocate?
    // The user asked for optimization.
    // Let's stick to Vec<Vec> for `hidden_matrix` to avoid complex index math during mutation (add_neuron),
    // OR just optimize the forward pass.
    // Providing a flattened structure is good for cache, but bad for "insert row/col".
    // Let's compromise: Flatten `input_matrix` is easy (width constant).
    // `hidden_matrix`: width changes.
    // Let's keep `hidden_matrix` as `Vec<Vec<f32>>` for now to make `add_neuron` readable,
    // removing `Rc` overhead is the big win.
    // WAIT, `input_matrix` width is constant (inputs+bias). So we can flatten that easily.

    // DECISION: To keep it idiomatic and readable (User: "optimize possibly bulky ... code"),
    // I will stick to `Vec<Vec<f32>>` because `add_neuron` inserts rows and columns.
    // Flattening a dynamic 2D array that grows in both dimensions is messy.
    // The main performance win comes from removing Rc<RefCell>.
    pub input_matrix: Vec<Vec<f32>>,
    pub hidden_matrix: Vec<Vec<f32>>,

    // for visualization/debug (like Python)
    pub last_inputs: Vec<f32>,
    pub last_activations: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new<R: Rng>(
        num_inputs: usize,
        num_outputs: usize,
        mutate: usize,
        bias: f32,
        rng: &mut R,
    ) -> Self {
        let neuron_number = num_inputs + 1 + num_outputs;

        // Initial setup
        let mut nn = Self {
            num_inputs,
            num_outputs,
            bias,
            neuron_number,
            eval_order: Vec::new(),
            // inputs+bias rows, initially num_outputs rows
            input_matrix: vec![vec![0.0; num_inputs + 1]; num_outputs],
            // hidden x hidden (initially outputs x outputs, zeroed)
            hidden_matrix: vec![vec![0.0; num_outputs]; num_outputs],
            last_inputs: Vec::new(),
            last_activations: Vec::new(),
        };

        for _ in 0..mutate {
            nn.mutate(rng);
        }

        nn
    }

    pub fn add_neuron(&mut self) {
        // Input matrix: add new zero row (new hidden neuron connection from inputs)
        self.input_matrix.push(vec![0.0_f32; self.num_inputs + 1]);

        // Hidden matrix: add new row and column
        // Add column to existing rows
        for row in &mut self.hidden_matrix {
            row.push(0.0_f32);
        }
        // Add new row (width = old width + 1)
        let new_width = self.hidden_matrix.len() + 1; // since we just pushed to existing rows
        self.hidden_matrix.push(vec![0.0_f32; new_width]);

        self.neuron_number += 1;
    }

    pub fn add_connection<R: Rng>(
        &mut self,
        source_id: usize,
        target_id: usize,
        weight: Option<f32>,
        rng: &mut R,
    ) {
        let w = weight.unwrap_or_else(|| rng.gen_range(-1.0..1.0));
        let in_bi = self.num_inputs + 1;

        if source_id < in_bi {
            if let Some(row) = self.input_matrix.get_mut(target_id - in_bi) {
                if let Some(val) = row.get_mut(source_id) {
                    *val = w;
                }
            }
        } else {
            if let Some(row) = self.hidden_matrix.get_mut(target_id - in_bi) {
                if let Some(val) = row.get_mut(source_id - in_bi) {
                    *val = w;
                }
            }
        }
    }

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
                    *w_ref += if rng.gen_bool(0.5) { -0.1 } else { 0.1 };
                    break;
                }
            }
        }
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
