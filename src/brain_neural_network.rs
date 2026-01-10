use crate::settings;
use rand::Rng;
// use std::f32::consts::PI;

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

// Python's round() is ties-to-even. We replicate for the one place it matters.
pub fn round_ties_even_to_i32(x: f32) -> i32 {
    let xf = x as f64;
    let floor = xf.floor();
    let diff = xf - floor;

    let eps = 1e-12;
    if diff < 0.5 - eps {
        floor as i32
    } else if diff > 0.5 + eps {
        (floor + 1.0) as i32
    } else {
        // tie: choose even
        let a = floor as i64;
        let b = a + 1;
        if a % 2 == 0 {
            a as i32
        } else {
            b as i32
        }
    }
}

// ----- Neural Network -----
#[derive(Clone)]
pub struct NeuralNetwork {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub bias: f32,
    pub neuron_number: usize, // total neurons: inputs + bias + outputs + hidden

    // topological order for hidden nodes (indices in activations vector)
    pub reihnfolge: Vec<usize>,

    // (outputs+hidden) x (inputs + bias)
    pub input_matrix: Vec<Vec<f32>>,

    // (outputs+hidden) x (outputs+hidden)
    pub hidden_matrix: Vec<Vec<f32>>,

    // for visualization/debug (like Python)
    pub last_inputs: Option<Vec<f32>>,
    pub last_activations: Option<Vec<f32>>,
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

        let mut nn = Self {
            num_inputs,
            num_outputs,
            bias,
            neuron_number,
            reihnfolge: Vec::new(),
            input_matrix: vec![vec![0.0; num_inputs + 1]; num_outputs],
            hidden_matrix: vec![vec![0.0; num_outputs]; num_outputs],
            last_inputs: None,
            last_activations: None,
        };

        for _ in 0..mutate {
            nn.mutate(rng);
        }

        nn
    }

    pub fn add_neuron(&mut self) {
        // Input matrix: add new zero row
        self.input_matrix.push(vec![0.0; self.num_inputs + 1]);

        // Hidden matrix: add new row and column
        let old = self.hidden_matrix.len();
        self.hidden_matrix.push(vec![0.0; old]);
        for row in &mut self.hidden_matrix {
            row.push(0.0);
        }

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
            self.input_matrix[target_id - in_bi][source_id] = w;
        } else {
            self.hidden_matrix[target_id - in_bi][source_id - in_bi] = w;
        }
    }

    fn mat_vec_mul(mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; mat.len()];
        for (i, row) in mat.iter().enumerate() {
            let mut sum = 0.0;
            for (j, w) in row.iter().enumerate() {
                sum += *w * vec[j];
            }
            out[i] = sum;
        }
        out
    }

    fn row_dot(row: &[f32], vec: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (w, v) in row.iter().zip(vec.iter()) {
            sum += *w * *v;
        }
        sum
    }

    pub fn forward_vectorized(&mut self, inputs: &[f32], energy: f32) -> [f32; 2] {
        // activations for inputs + bias
        let mut in_act = vec![0.0; self.num_inputs + 1];
        for i in 0..self.num_inputs {
            in_act[i] = inputs[i];
        }
        in_act[self.num_inputs] = self.bias * energy;

        // initial activations (outputs + hidden)
        let mut activations = Self::mat_vec_mul(&self.input_matrix, &in_act);

        // hidden evaluation in topological order
        for &order in &self.reihnfolge {
            let dot = Self::row_dot(&self.hidden_matrix[order], &activations);
            activations[order] = act_func(activations[order] + dot);
        }

        // store for debug/visualization
        self.last_inputs = Some(inputs.to_vec());
        self.last_activations = Some(activations.clone());

        // outputs are indices 0 and 1
        let out0 = act_speed(activations[0] + Self::row_dot(&self.hidden_matrix[0], &activations));
        let out1 = act_angle(activations[1] + Self::row_dot(&self.hidden_matrix[1], &activations));

        [out0, out1]
    }

    pub fn mutate<R: Rng>(&mut self, rng: &mut R) {
        let in_bi = self.num_inputs + 1;

        if rng.gen::<f32>() < settings::add_neuron() {
            // Collect existing connections
            // Format: (source_id, target_id, weight)
            let mut connections = Vec::new();

            // From input matrix (inputs+bias -> outputs+hidden)
            for r in 0..self.input_matrix.len() {
                for c in 0..self.input_matrix[r].len() {
                    let w = self.input_matrix[r][c];
                    if w != 0.0 {
                        let target = in_bi + r;
                        let source = c;
                        connections.push((source, target, w));
                    }
                }
            }

            // From hidden matrix (hidden -> outputs+hidden)
            for r in 0..self.hidden_matrix.len() {
                for c in 0..self.hidden_matrix[r].len() {
                    let w = self.hidden_matrix[r][c];
                    if w != 0.0 {
                        let target = in_bi + r;
                        let source = in_bi + c;
                        connections.push((source, target, w));
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

                // Add Source -> New (weight = old weight)
                self.add_connection(source, new_neuron_id, Some(weight), rng);

                // Add New -> Target (weight = old weight)
                self.add_connection(new_neuron_id, target, Some(weight), rng);
            }
        }

        if rng.gen::<f32>() < settings::add_weight() {
            let mut sources: Vec<usize> = (0..in_bi).collect();

            sources.extend((in_bi + self.num_outputs)..self.neuron_number); // hidden only (no outputs)

            let source = sources[rng.gen_range(0..sources.len())];
            let target = rng.gen_range(in_bi..self.neuron_number);

            let mut single_connectn = true;
            if source < in_bi {
                if self.input_matrix[target - in_bi][source] != 0.0 {
                    single_connectn = false;
                }
            } else if self.hidden_matrix[target - in_bi][source - in_bi] != 0.0 {
                single_connectn = false;
            }

            if single_connectn {
                if !self.infinity_loop(source, target) {
                    self.add_connection(source, target, None, rng);
                }
            }
        }

        if rng.gen::<f32>() < settings::change_weight() {
            for _ in 0..4 {
                let mut sources: Vec<usize> = (0..in_bi).collect();
                sources.extend((in_bi + self.num_outputs)..self.neuron_number);

                let source = sources[rng.gen_range(0..sources.len())];
                let target = rng.gen_range(in_bi..self.neuron_number);

                if source < in_bi {
                    let w = &mut self.input_matrix[target - in_bi][source];
                    if *w != 0.0 {
                        *w += if rng.gen_bool(0.5) { -0.1 } else { 0.1 };
                        break;
                    }
                } else {
                    let w = &mut self.hidden_matrix[target - in_bi][source - in_bi];
                    if *w != 0.0 {
                        *w += if rng.gen_bool(0.5) { -0.1 } else { 0.1 };
                        break;
                    }
                }
            }
        }
    }

    // Python: infinityLoop(source, target)
    pub fn infinity_loop(&mut self, source: usize, target: usize) -> bool {
        let in_bi = self.num_inputs + 1;
        let in_bi_out = in_bi + self.num_outputs;

        // only if BOTH are hidden ids
        if source >= in_bi_out && target >= in_bi_out {
            let ot_hi = self.neuron_number - in_bi; // outputs + hidden count

            // dummy insert
            self.hidden_matrix[target - in_bi][source - in_bi] = 1.0;

            let mut sample_reihnfolge: Vec<usize> = Vec::new();
            let mut index_list: Vec<usize> = (self.num_outputs..ot_hi).collect(); // hidden indices only

            // remove "null rows" (no incoming edges)
            let mut k = 0;
            while k < index_list.len() {
                let i = index_list[k];
                if self.hidden_matrix[i].iter().all(|&v| v == 0.0) {
                    index_list.remove(k);
                    sample_reihnfolge.push(i);
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
                    let no_incoming_from_remaining = index_list
                        .iter()
                        .all(|&col| self.hidden_matrix[i][col] == 0.0);

                    if no_incoming_from_remaining {
                        index_list.remove(j);
                        sample_reihnfolge.push(i);
                    } else {
                        j += 1;
                    }
                }

                if before == index_list.len() {
                    // cycle -> revert dummy
                    self.hidden_matrix[target - in_bi][source - in_bi] = 0.0;
                    return true;
                }
            }

            self.reihnfolge = sample_reihnfolge;
            false
        } else {
            false
        }
    }

    // ---- Visualization-ish: write DOT (GraphViz) like Python plot_neural_network ----
    pub fn to_dot(&self) -> String {
        let num_inputs = self.num_inputs;
        let num_outputs = self.num_outputs;
        let total_neurons = self.neuron_number;
        let num_hidden = total_neurons - (num_inputs + 1 + num_outputs);

        let input_nodes: Vec<String> = (0..num_inputs).map(|i| format!("Input {}", i)).collect();
        let bias_node = vec!["Bias".to_string()];
        let output_nodes: Vec<String> = (0..num_outputs).map(|i| format!("Output {}", i)).collect();
        let hidden_nodes: Vec<String> = (0..num_hidden).map(|i| format!("Hidden {}", i)).collect();

        let mut all_nodes = Vec::new();
        all_nodes.extend(input_nodes);
        all_nodes.extend(bias_node);
        all_nodes.extend(output_nodes);
        all_nodes.extend(hidden_nodes);

        let mut dot = String::new();
        dot.push_str("digraph NeuralNetwork {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=circle];\n");

        // nodes
        for n in &all_nodes {
            dot.push_str(&format!("  \"{}\";\n", n));
        }

        // edges from input_matrix
        let ot_hi = num_outputs + num_hidden;
        for i in 0..ot_hi {
            for j in 0..(num_inputs + 1) {
                let w = self.input_matrix[i][j];
                if w != 0.0 {
                    let src = &all_nodes[j];
                    let dst = &all_nodes[num_inputs + 1 + i];
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [label=\"{:.2}\"];\n",
                        src, dst, w
                    ));
                }
            }
        }

        // edges from hidden_matrix
        for i in 0..ot_hi {
            for j in 0..ot_hi {
                let w = self.hidden_matrix[i][j];
                if w != 0.0 {
                    let src = &all_nodes[num_inputs + 1 + j];
                    let dst = &all_nodes[num_inputs + 1 + i];
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [label=\"{:.2}\"];\n",
                        src, dst, w
                    ));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }
}
