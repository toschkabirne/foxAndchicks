// ============================================================================
// IMPORTS AND DEPENDENCIES
// ============================================================================

use crate::brain_neural_network::NeuralNetwork;
use crate::settings::*;
use crate::spatial_hash::HasPos;
use ::rand::Rng;
use macroquad::prelude::*;
use std::iter::IntoIterator;

use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// GLOBAL STATE AND CONSTANTS
// ============================================================================

/// Global counter for assigning unique IDs to animals.
///
/// Design rationale: We use `AtomicUsize` to ensure thread-safe ID generation
/// even though the current game loop is single-threaded. This prevents subtle
/// bugs if we later add multi-threading (e.g., for parallel simulation steps).
/// The atomic operations have negligible overhead compared to the benefit of
/// future-proofing the codebase.
static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// Pre-calculated constant for 2π (full circle in radians).
///
/// Design rationale: We compute this at compile time to avoid repeated
/// calculations during runtime. This is used frequently in angle normalization
/// and prey's 360° vision calculation, so caching it improves performance.
const TWO_PI: f32 = 2.0 * PI;

// ============================================================================
// HELPER FUNCTIONS - ID GENERATION
// ============================================================================

/// Generates a unique ID for each animal.
///
/// Design rationale: Using a centralized ID generator ensures each animal
/// (predator or prey) has a unique identifier for tracking during hunting,
/// reproduction, and data collection. We use `Ordering::Relaxed` because
/// we only need atomicity, not ordering guarantees (single-threaded context).
#[inline]
fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// HELPER FUNCTIONS - TOROIDAL WORLD GEOMETRY
// ============================================================================
// Design rationale for toroidal world:
// We implement a "wrap-around" world where entities crossing one edge appear
// on the opposite edge. This eliminates edge effects where animals could get
// "stuck" in corners, creating a more uniform simulation environment. This
// is especially important for evolution simulations where environmental biases
// should be minimized.
// ============================================================================

/// Wraps a position to stay within world boundaries (toroidal topology).
///
/// Takes a position that may be outside [0, width) × [0, height) and maps it
/// back into bounds using modular arithmetic.
///
/// Design rationale: We use `rem_euclid` instead of the `%` operator because
/// it handles negative numbers correctly (e.g., -1.0 wraps to width-1.0).
/// This is critical for smooth movement across boundaries.
///
/// # Arguments
/// * `pos` - The position to wrap (can be outside bounds)
/// * `width` - World width
/// * `height` - World height
#[inline]
pub fn wrap_position(pos: Vec2, width: f32, height: f32) -> Vec2 {
    vec2(pos.x.rem_euclid(width), pos.y.rem_euclid(height))
}

/// Computes the shortest vector from point `a` to point `b` in a toroidal world.
///
/// In a toroidal world, there are multiple paths between any two points. For
/// example, going left or going right (wrapping around). This function always
/// returns the shortest path.
///
/// Design rationale: This is crucial for:
/// 1. Predator vision: A predator near the right edge should "see" prey on
///    the left edge as being nearby, not far away.
/// 2. Hunting logic: Predators should chase prey via the shortest path.
/// 3. Angle calculations: Animals should turn toward the nearest instance of
///    their target, accounting for world wrapping.
///
/// The algorithm works by checking if the direct path is longer than half the
/// world size. If so, going the "other way" (across the boundary) is shorter.
///
/// # Returns
/// A vector pointing from `a` to `b`, with components in range:
/// * dx ∈ [-width/2, width/2]
/// * dy ∈ [-height/2, height/2]
#[inline]
pub fn wrapped_distance_vector(a: Vec2, b: Vec2, width: f32, height: f32) -> Vec2 {
    let mut dx = b.x - a.x;
    let mut dy = b.y - a.y;

    // If the direct distance is more than half the world width,
    // the wrapped distance (going the other way) is shorter
    if dx > width / 2.0 {
        dx -= width; // Wrap left
    } else if dx < -width / 2.0 {
        dx += width; // Wrap right
    }

    // Same logic for vertical distance
    if dy > height / 2.0 {
        dy -= height; // Wrap up
    } else if dy < -height / 2.0 {
        dy += height; // Wrap down
    }

    vec2(dx, dy)
}

/// Computes the shortest Euclidean distance between two points in a toroidal world.
///
/// This is simply the length of the shortest vector from `a` to `b`.
///
/// Design rationale: Used for collision detection (predator eating prey) and
/// range checks (is something within vision range?). By using the wrapped
/// distance, animals can interact across world boundaries naturally.
#[inline]
pub fn wrapped_distance_abs(a: Vec2, b: Vec2, width: f32, height: f32) -> f32 {
    wrapped_distance_vector(a, b, width, height).length()
}

// ============================================================================
// HELPER FUNCTIONS - ANGLE MANIPULATION
// ============================================================================

/// Normalizes an angle to the range [-π, π].
///
/// Design rationale: Angles can accumulate over time (e.g., an animal turning
/// continuously in one direction). Without normalization, angles would grow
/// unbounded, potentially causing:
/// 1. Numerical precision issues with large float values
/// 2. Difficulties comparing angles (is 7π the same direction as π?)
/// 3. Wraparound bugs in angle calculations
///
/// We use [-π, π] instead of [0, 2π] because it makes angular differences
/// easier to interpret: a difference of 0.1 means "turn slightly right",
/// regardless of absolute heading.
///
/// # Algorithm
/// 1. Add π to shift range from [-π, π] to [0, 2π]
/// 2. Apply modulo 2π to wrap to [0, 2π]
/// 3. Subtract π to shift back to [-π, π]
#[inline]
pub fn normalize_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TWO_PI) - PI
}

/// Linearly interpolates between two angles, accounting for wraparound.
///
/// Design rationale: Used for predator vision rays. When a predator looks
/// in a cone from angle A to angle B, we need to place vision rays evenly
/// between them. However, naive interpolation fails when angles wrap around:
/// e.g., lerp(170°, -170°, 0.5) should give 180°, not 0°.
///
/// This function handles wraparound by:
/// 1. Computing the shortest angular difference from `a` to `b`
/// 2. Interpolating along that shortest path
/// 3. Normalizing the result
///
/// # Arguments
/// * `a` - Start angle
/// * `b` - End angle  
/// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
#[inline]
fn angle_lerp(a: f32, b: f32, t: f32) -> f32 {
    let d = normalize_angle(b - a); // Shortest angular path
    normalize_angle(a + d * t)
}

// ============================================================================
// HELPER FUNCTIONS - VISUALIZATION
// ============================================================================

/// Draws a line that wraps around toroidal world boundaries.
///
/// Design rationale: When visualizing animal vision rays, a ray might extend
/// beyond the world boundary. To properly show the toroidal nature of the world,
/// we need to draw the "continuation" of the line on the opposite side.
///
/// For example, if a predator at x=50 looks left with a vision ray extending
/// to x=-100, we need to:
/// 1. Draw the ray from x=50 to x=0 (normal)
/// 2. Draw the continuation from x=width to x=width-100 (wrapped portion)
///
/// This makes it visually clear that animals can see across boundaries.
///
/// # Algorithm
/// 1. Always draw the main line (may extend outside bounds)
/// 2. Check if the endpoint is outside boundaries
/// 3. For each boundary crossed, draw an offset copy of the line
/// 4. Handle corner cases where both x and y boundaries are crossed
fn draw_wrapped_line(
    start: Vec2,
    end: Vec2,
    width: f32,
    height: f32,
    thickness: f32,
    color: Color,
) {
    // Always draw the main line
    draw_line(start.x, start.y, end.x, end.y, thickness, color);

    // Determine which boundaries (if any) are crossed
    let mut offsets = Vec::new();

    if end.x < 0.0 {
        offsets.push(vec2(width, 0.0)); // Line exits left, draw copy on right
    } else if end.x >= width {
        offsets.push(vec2(-width, 0.0)); // Line exits right, draw copy on left
    }

    if end.y < 0.0 {
        offsets.push(vec2(0.0, height)); // Line exits top, draw copy on bottom
    } else if end.y >= height {
        offsets.push(vec2(0.0, -height)); // Line exits bottom, draw copy on top
    }

    // Draw wrapped segments for each boundary crossing
    for offset in &offsets {
        draw_line(
            start.x + offset.x,
            start.y + offset.y,
            end.x + offset.x,
            end.y + offset.y,
            thickness,
            color,
        );
    }

    // Corner case: line crosses both x and y boundaries
    // Need to draw a copy at the diagonal opposite corner
    if offsets.len() == 2 {
        let corner_offset = offsets[0] + offsets[1];
        draw_line(
            start.x + corner_offset.x,
            start.y + corner_offset.y,
            end.x + corner_offset.x,
            end.y + corner_offset.y,
            thickness,
            color,
        );
    }
}

// ============================================================================
// SHARED ANIMAL CORE
// ============================================================================
// Design rationale for AnimalCore:
// Both Predator and Prey share common attributes (position, angle, energy,
// brain). Instead of duplicating these fields and their methods, we extract
// them into a shared struct. This follows the DRY principle and ensures
// consistency in how core properties are accessed and modified.
//
// Why not use inheritance/traits?
// Rust doesn't have classical inheritance. While we could use composition
// with traits, the simple struct approach is more idiomatic for data that
// truly is shared. Each animal type (Predator/Prey) then adds its specific
// fields (e.g., eaten_prey, rest_time) as needed.
// ============================================================================

/// Core data shared by all animals (Predators and Prey).
///
/// This struct contains the fundamental state that every animal in the
/// simulation needs, regardless of its specific role.
#[derive(Clone)]
pub struct AnimalCore {
    /// Unique identifier for this animal (never reused, even after death)
    pub id: usize,

    /// Current 2D position in the world
    pub pos: Vec2,

    /// Current heading angle in radians (0 = facing right/east)
    pub angle: f32,

    /// Current energy level (animal dies/cannot reproduce when too low)
    pub energy: f32,

    /// Neural network brain that controls decision-making
    ///
    /// Design rationale: Each animal owns its brain (no Rc<RefCell<>>).
    /// This was refactored from a shared pointer approach to improve:
    /// 1. Memory safety: No risk of multiple mutable borrows
    /// 2. Performance: No runtime borrow checking overhead
    /// 3. Clarity: Each animal clearly owns and controls its own brain
    /// 4. Evolution: Mutations create new brain instances rather than
    ///    modifying shared state
    pub brain: NeuralNetwork,
}

impl AnimalCore {
    /// Creates a new AnimalCore with an existing brain.
    ///
    /// This is used during reproduction where the child inherits a mutated
    /// copy of the parent's brain.
    pub fn new_with_brain(pos: Vec2, angle: f32, energy: f32, brain: NeuralNetwork) -> Self {
        Self {
            id: next_id(),
            pos,
            angle,
            energy,
            brain,
        }
    }

    /// Returns the x-coordinate of the animal's position.
    ///
    /// Design rationale: These getters provide a consistent interface for
    /// position access, used by the spatial hash for efficient neighbor queries.
    #[inline]
    pub fn x(&self) -> f32 {
        self.pos.x
    }

    /// Returns the y-coordinate of the animal's position.
    #[inline]
    pub fn y(&self) -> f32 {
        self.pos.y
    }

    /// Sets a new position for the animal.
    ///
    /// Design rationale: The spatial hash uses this to update positions
    /// after movement.
    #[inline]
    pub fn set_xy(&mut self, x: f32, y: f32) {
        self.pos = vec2(x, y);
    }
}

/// Creates a child brain from a parent brain with random mutations.
///
/// Design rationale: This is the core of the evolutionary mechanism.
/// When an animal reproduces, its child inherits the parent's brain structure
/// and weights, but with small random changes (mutations). Over generations,
/// beneficial mutations lead to better survival and more offspring.
///
/// Why multiple mutations (k = 2..=6)?
/// - Too few mutations: Evolution is extremely slow
/// - Too many mutations: Children are too different from successful parents,
///   losing beneficial traits
/// - Random range: Adds variation in mutation rate, allowing the population
///   to explore the fitness landscape at different speeds
///
/// # Arguments
/// * `parent` - The parent's neural network to inherit from
/// * `rng` - Random number generator for mutation randomness
#[inline]
fn inherited_brain_with_mutations<R: Rng>(parent: &NeuralNetwork, rng: &mut R) -> NeuralNetwork {
    let mut brain = parent.clone();
    let k = rng.gen_range(2..=6); // Apply multiple mutations
    for _ in 0..k {
        brain.mutate(rng);
    }
    brain
}

/// Performs a movement step for an animal, updating position, angle, and energy.
///
/// This is shared movement logic used by both Predators and Prey.
///
/// Design rationale:
/// 1. **Quadratic energy cost**: Energy cost is proportional to speed_factor²,
///    not speed_factor. This creates a strong selective pressure for efficiency:
///    - Moving at half speed costs 1/4 the energy
///    - Animals must balance speed vs. energy conservation
///    - Encourages prey to "rest" when safe, predators to hunt strategically
///
/// 2. **Energy capping**: We prevent energy from exceeding max_energy to avoid
///    exploits where animals could accumulate unlimited energy through resting
///    or eating. This keeps energy as a meaningful constraint.
///
/// 3. **Separated turning and movement**: Animals first turn, then move forward.
///    This matches how real organisms typically orient before moving, and makes
///    the neural network's task clearer (one output for direction, one for speed).
///
/// # Arguments
/// * `core` - The animal's core state (position, angle, energy)
/// * `speed_factor` - Multiplier for movement (0.0 = stationary, 1.0 = full speed)
/// * `turn_delta` - Change in angle this frame (in radians)
/// * `speed` - Base movement speed for this animal type
/// * `moving_decay` - Energy cost multiplier for movement
/// * `max_energy` - Maximum energy cap for this animal type
#[inline]
fn move_with_speed_factor(
    core: &mut AnimalCore,
    speed_factor: f32,
    turn_delta: f32,
    speed: f32,
    moving_decay: f32,
    max_energy: f32,
) {
    // Apply turning
    core.angle = normalize_angle(core.angle + turn_delta);

    // Apply forward movement based on current heading
    core.pos.x += speed_factor * speed * core.angle.cos();
    core.pos.y += speed_factor * speed * core.angle.sin();

    // Handle world wrapping (toroidal topology)
    core.pos = wrap_position(core.pos, SCREEN_WIDTH as f32, SCREEN_HEIGHT as f32);

    // Deduct energy cost (quadratic in speed for realism)
    core.energy -= (speed_factor * speed_factor) * moving_decay;

    // Enforce energy cap to prevent accumulation exploits
    if core.energy > max_energy {
        core.energy = max_energy;
    }
}

// ============================================================================
// PREDATOR IMPLEMENTATION
// ============================================================================
// Design rationale for Predator:
// Predators are the "hunters" in the simulation. They must:
// 1. Detect prey using limited forward-facing vision (realistic constraint)
// 2. Chase and catch prey (collision detection)
// 3. Manage energy carefully (hunting costs energy, eating restores it)
// 4. Reproduce when successful (evolutionary reward for good hunters)
//
// The selective pressure on predators:
// - Must balance aggressive hunting (high energy cost) vs. conservation
// - Must develop effective vision-based tracking behaviors
// - Population naturally limits itself (no prey = predators starve)
// ============================================================================

/// A predator animal that hunts prey.
#[derive(Clone)]
pub struct Predator {
    /// Core animal state (position, angle, energy, brain)
    pub core: AnimalCore,

    /// Counter for prey eaten since last reproduction.
    ///
    /// Design rationale: This implements a "reproduction threshold" mechanic.
    /// Predators must catch 3 prey to reproduce, which:
    /// 1. Prevents exponential predator growth (would crash the ecosystem)
    /// 2. Creates selective pressure for hunting skill
    /// 3. Ties reproduction to actual survival fitness (catching prey)
    /// 4. Naturally limits predator population to prey availability
    pub eaten_prey: i32,

    /// Cooldown timer preventing immediate re-reproduction.
    ///
    /// Design rationale: After reproducing, a predator cannot reproduce again
    /// for several frames, even if it catches 3 more prey. This prevents:
    /// 1. Population explosion in prey-rich areas
    /// 2. Unrealistic "rapid-fire" reproduction
    /// 3. Spatial clustering of predators (they need time to spread)
    ///
    /// The German comment "frames bis fressen wieder erlaubt" confirms this
    /// is a frame-based cooldown.
    pub repro_cooldown: i32,
}

/// Trait implementation for spatial hash queries.
///
/// Design rationale: The spatial hash accelerates "find nearby entities" queries
/// from O(n) to O(1) average case. This is critical for performance when checking:
/// - What prey can thisпредаtor see?
/// - What predators can this prey see?
/// - Is this predator close enough to eat this prey?
impl HasPos for Predator {
    fn x(&self) -> f32 {
        self.core.x()
    }
    fn y(&self) -> f32 {
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.core.set_xy(x, y);
    }
}

impl Predator {
    /// Creates a new predator with a random brain and position.
    ///
    /// Design rationale: Each predator starts with:
    /// 1. Random heading: Prevents all predators moving in the same direction
    /// 2. Fresh neural network: Initialized with `pred_init_mut()` mutation rate
    /// 3. Full energy: Gives new predators a fair chance to hunt
    /// 4. Zero kill count: Must prove itself by hunting
    ///
    /// # Arguments
    /// * `x`, `y` - Starting position
    /// * `rng` - Random number generator
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..TWO_PI);
        // Create brain with predator-specific input count and mutation rate
        let brain = NeuralNetwork::new(NUMBER_SIGHTS_PREDATOR, 2, pred_init_mut(), bias(), rng);

        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PRED_ENERGY, brain),
            eaten_prey: 0,
            repro_cooldown: 0,
        }
    }

    /// Returns the unique ID of this predator.
    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Computes sensory inputs for the predator's neural network.
    ///
    /// Predators have forward-facing "cone vision" covering ±30° from their heading.
    /// The cone is divided into NUMBER_SIGHTS_PREDATOR rays, each acting as a
    /// boolean "prey detector" (1.0 if prey detected, 0.0 otherwise).
    ///
    /// Design rationale:
    /// 1. **Limited field of view (60° total)**: Predators must turn to see around
    ///    them, creating more realistic hunting behavior. Prey behind a predator
    ///    are safe (for now).
    ///
    /// 2. **Angular width calculation**: We don't just check if prey is on a ray.
    ///    Instead, we calculate the angular size of the prey "disc" at that distance
    ///    and check if it overlaps with the ray. This simulates realistic vision:
    ///    - Close prey are easier to see (larger angular size)
    ///    - Distant prey are harder to detect (smaller angular size)
    ///    - Creates smooth activation as prey move relative to vision rays
    ///
    /// 3. **Range limit (SIGHT_RANGE_PREDATOR)**: Predators can't see infinitely
    ///    far, creating strategic depth (must get close enough to see prey).
    ///
    /// 4. **Toroidal distance**: Uses wrapped_distance_vector so predators can
    ///    see prey across world boundaries.
    ///
    /// 5. **Early activation**: Once a ray detects prey (inputs[i] = 1.0), we
    ///    skip checking more prey for that ray (optimization + "this ray is blocked").
    ///
    /// # Algorithm for angular width:
    /// If prey has radius R and is at distance D:
    /// - Angular width ≈ 2 * arcsin(R/D)
    /// - We use arcsin(R/D) as half-width and check if abs(ray_angle - prey_angle) < half-width
    /// - This is a small-angle approximation but works well for this simulation
    ///
    /// # Arguments
    /// * `preys` - Iterator over prey to check for visibility
    ///
    /// # Returns
    /// Vector of floats (0.0 or 1.0) representing each vision ray
    pub fn get_inputs<'a, I>(&self, preys: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Prey>,
    {
        let n = NUMBER_SIGHTS_PREDATOR.max(1);
        let mut inputs = vec![0.0; n];

        // Define the vision cone boundaries
        let start_angle = normalize_angle(self.core.angle - 30.0_f32.to_radians());
        let end_angle = normalize_angle(self.core.angle + 30.0_f32.to_radians());

        let predator_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for prey in preys {
            let prey_pos = prey.core.pos;

            // Compute shortest vector to prey (accounting for world wrapping)
            let delta = wrapped_distance_vector(predator_pos, prey_pos, world_w, world_h);
            let dist = delta.length();

            // Skip if prey is too close (div-by-zero) or out of range
            if dist <= 0.0 || dist >= SIGHT_RANGE_PREDATOR {
                continue;
            }

            // Compute angle to prey using wrapped delta
            let angle_to_prey = delta.y.atan2(delta.x);

            // Calculate angular width of prey "disc" at this distance
            // This is the half-angle subtended by the prey's radius
            let mut val = PREY_RADIUS / dist;
            if val > 1.0 {
                val = 1.0; // Clamp for arcsin domain
            }
            let angular_width = val.asin();

            // Check each vision ray to see if prey overlaps with it
            for i in 0..n {
                if inputs[i] == 1.0 {
                    continue; // This ray already detected prey
                }

                // Interpolate angle for this ray within the vision cone
                let t = if n > 1 {
                    i as f32 / (n as f32 - 1.0) // Map index to [0, 1]
                } else {
                    0.0 // Single ray: use middle of cone
                };
                let ray_angle = angle_lerp(start_angle, end_angle, t);

                // Check if prey's angular disc overlaps this ray
                let diff = normalize_angle(ray_angle - angle_to_prey).abs();
                if diff < angular_width {
                    inputs[i] = 1.0; // Prey detected on this ray!
                }
            }
        }

        inputs
    }

    /// Executes one movement/decision step for the predator.
    ///
    /// Design rationale:
    /// 1. **Passive energy decay**: Predators lose energy each frame just for
    ///    existing (PRED_DEFAULT_DECAY). This creates time pressure - can't wait
    ///    forever for prey to appear.
    ///
    /// 2. **Energy ratio input**: The brain receives energy_ratio (current/max)
    ///    as a bias input. This allows the brain to "know" if it's low on energy
    ///    and should hunt more aggressively or conserve energy.
    ///
    /// 3. **Output clamping**:
    ///    - speed_factor ∈ [0, 1]: Can't move backward or faster than max
    ///    - turn_delta ∈ [-π/2, π/2]: Maximum 90° turn per frame prevents
    ///      unrealistic instant-180° turns
    ///
    /// 4. **Quadratic movement cost**: See move_with_speed_factor docs.
    ///    This is applied IN ADDITION to the passive decay, so moving is expensive.
    ///
    /// # Arguments
    /// * `inputs` - Sensory inputs from get_inputs (vision rays)
    pub fn move_step(&mut self, inputs: &[f32]) {
        // Passive energy decay (cost of living)
        self.core.energy -= PRED_DEFAULT_DECAY;

        // Compute energy ratio for brain decision-making
        let energy_ratio = self.core.energy / PRED_ENERGY;

        // Run neural network to get movement decisions
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        // Extract and clamp movement parameters
        let speed_factor = outputs[0].clamp(0.0, 1.0);
        let turn_delta = outputs[1].clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_2;

        // Apply movement (includes additional energy cost)
        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PREDATOR_SPEED,
            PRED_MOVING_DECAY,
            PRED_ENERGY,
        );
    }

    /// Checks for nearby prey and attempts to eat them.
    ///
    /// This is the "hunting" phase where predators can catch and eat prey they've
    /// gotten close enough to.
    ///
    /// Design rationale:
    /// 1. **Collision radius**: The eating distance is the sum of both radii
    ///    (PREDATOR_RADIUS + PREY_RADIUS). This means predators need to actually
    ///    "touch" their prey to catch them.
    ///
    /// 2. **HashSet deduplication**: Multiple predators might try to eat the same
    ///    prey in a single frame. The `eaten_prey_ids` HashSet ensures each prey
    ///    can only be eaten once per frame, preventing:
    ///    - Double-counting kills
    ///    - Multiple energy gains from one prey
    ///    - Allowing predators to "share" kills (first to touch it wins)
    ///
    /// 3. **Energy capped at max**: Energy gain is added but capped at PRED_ENERGY.
    ///    This prevents predators from accumulating unlimited energy reserves.
    ///
    /// 4. **Automatic reproduction attempt**: After each kill, we immediately check
    ///    if the predator can reproduce. This couples eating success directly to
    ///    reproduction, creating strong evolutionary pressure.
    ///
    /// 5. **Multiple kills allowed**: The commented-out `break` shows we considered
    ///    limiting to one kill per frame, but decided to allow multiple. This lets
    ///    predators feeding on dense prey populations eat more efficiently.
    ///
    /// 6. **Toroidal collision detection**: Uses wrapped_distance_abs so predators
    ///    can catch prey across world boundaries.
    ///
    /// # Arguments
    /// * `prey_candidates` - Nearby prey (typically from spatial hash query)
    /// * `eaten_prey_ids` - Set to track which prey have been eaten this frame
    /// * `newborn_preds` - Vec to collect any offspring produced
    /// * `rng` - Random number generator for reproduction
    pub fn hunt_nearby<'a, R: Rng, I>(
        &mut self,
        prey_candidates: I,
        eaten_prey_ids: &mut HashSet<usize>,
        newborn_preds: &mut Vec<Predator>,
        rng: &mut R,
    ) where
        I: IntoIterator<Item = &'a Prey>,
    {
        // Collision threshold: predator and prey radii combined
        let eat_r = PREDATOR_RADIUS + PREY_RADIUS;
        let predator_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for prey in prey_candidates {
            let id = prey.core.id;

            // Skip prey that have already been eaten this frame
            if eaten_prey_ids.contains(&id) {
                continue;
            }

            // Check if close enough to eat (using toroidal distance)
            let prey_pos = prey.core.pos;
            let dist = wrapped_distance_abs(predator_pos, prey_pos, world_w, world_h);

            if dist < eat_r {
                // Mark prey as eaten (prevents double-eating)
                eaten_prey_ids.insert(id);

                // Gain energy (capped at maximum)
                self.core.energy = (self.core.energy + PREDATOR_ENERGY_GAIN).min(PRED_ENERGY);

                // Increment kill counter
                self.eaten_prey += 1;

                // Attempt to reproduce (may succeed if threshold reached)
                if let Some(child) = self.reproduce(rng) {
                    newborn_preds.push(child);
                }

                // Allow multiple kills per frame (break to limit to one)
                // If you want "max one kill per predator per frame", uncomment:
                // break;
            }
        }
    }

    /// Attempts to reproduce, creating an offspring if conditions are met.
    ///
    /// Design rationale:
    /// 1. **Dual-gating reproduction**: Requires BOTH:
    ///    - Cooldown expired (repro_cooldown <= 0)
    ///    - Kill threshold met (eaten_prey >= 3)
    ///    
    ///    This prevents both rapid reproduction and reproduction without skill.
    ///
    /// 2. **Cooldown value (5 frames)**: After reproducing, must wait 5 frames.
    ///    At 60 FPS, this is ~0.08 seconds. This prevents predator explosions but
    ///    allows successful hunters to reproduce relatively quickly.
    ///
    /// 3. **Reset eaten_prey counter**: After reproducing, the counter resets to 0.
    ///    The predator must catch 3 MORE prey to reproduce again. This ensures
    ///    continued selective pressure even for established predators.
    ///
    /// 4. **Small spatial offset**: Child spawns ±1 pixel from parent. This is:
    ///    - Small enough that they're essentially at the same location
    ///    - Large enough that they don't have identical positions (useful for debugging)
    ///    - Prevents perfect stacking which could cause visual glitches
    ///
    /// 5. **Brain inheritance with mutation**: The child gets a mutated copy of
    ///    the parent's brain. This is the core of the genetic algorithm:
    ///    - Successful hunters pass on their neural network structure
    ///    - Mutations allow exploration of better strategies
    ///    - Over generations, hunting skills improve
    ///
    /// 6. **Full energy for offspring**: The child starts with full energy
    ///    (PRED_ENERGY), not a split of the parent's energy. This design choice:
    ///    - Doesn't punish the parent for reproducing (energy-wise)
    ///    - Gives the child a fair chance (not starting exhausted)
    ///    - Places the reproduction cost in the "3 kills" requirement, not energy
    ///
    /// # Returns
    /// `Some(Predator)` if reproduction successful, `None` otherwise
    pub fn reproduce<R: Rng>(&mut self, rng: &mut R) -> Option<Predator> {
        const REPRO_COOLDOWN_FRAMES: i32 = 5;

        // Check cooldown
        if self.repro_cooldown > 0 {
            return None;
        }

        // Check kill threshold
        if self.eaten_prey < 3 {
            return None;
        }

        // Reset counters for next reproduction cycle
        self.eaten_prey = 0;
        self.repro_cooldown = REPRO_COOLDOWN_FRAMES;

        // Spawn child near parent (tiny random offset)
        let ox = self.core.pos.x + rng.gen_range(-1..=1) as f32;
        let oy = self.core.pos.y + rng.gen_range(-1..=1) as f32;

        // Create child with inherited, mutated brain
        let mut child = Predator::new(ox, oy, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        // Child starts with full energy (not split from parent)
        child.core.energy = PRED_ENERGY;

        Some(child)
    }

    /// Draws the predator's vision cone for debugging/visualization.
    ///
    /// Design rationale: Visualizing vision rays helps us:
    /// 1. Debug vision system (are rays actually detecting prey?)
    /// 2. Understand predator behavior (what can they see?)
    /// 3. Demonstrate the toroidal world (rays wrap across boundaries)
    ///
    /// Uses draw_wrapped_line to properly show vision across world edges.
    pub fn draw_sight(&self) {
        let n = NUMBER_SIGHTS_PREDATOR.max(1);
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        // Same vision cone as in get_inputs
        let start_angle = self.core.angle - 30.0_f32.to_radians();
        let end_angle = self.core.angle + 30.0_f32.to_radians();

        // Draw each vision ray
        for i in 0..n {
            let t = if n > 1 {
                i as f32 / (n as f32 - 1.0)
            } else {
                0.0
            };
            let sight_angle = start_angle + t * (end_angle - start_angle);

            // Calculate ray endpoint
            let end_x = self.core.pos.x + SIGHT_RANGE_PREDATOR * sight_angle.cos();
            let end_y = self.core.pos.y + SIGHT_RANGE_PREDATOR * sight_angle.sin();

            // Draw with wrapping support
            draw_wrapped_line(
                self.core.pos,
                vec2(end_x, end_y),
                world_w,
                world_h,
                1.0,
                YELLOW,
            );
        }
    }

    /// Draws the predator as a red circle with vision rays.
    ///
    /// Design rationale: Red color clearly distinguishes predators from prey
    /// (which are green), making population dynamics visible at a glance.
    pub fn draw(&self) {
        draw_circle(self.core.pos.x, self.core.pos.y, PREDATOR_RADIUS, RED);
        self.draw_sight();
    }
}

// ============================================================================
// PREY IMPLEMENTATION
// ============================================================================
// Design rationale for Prey:
// Prey are the "hunted" in the simulation. They must:
// 1. Detect predators using 360° vision (can see in all directions)
// 2. Evade predators through movement (run away)
// 3. Manage energy carefully (fleeing costs energy, resting recovers it)
// 4. Reproduce over time (population growth)
//
// The selective pressure on prey:
// - Must develop effective predator-avoidance behaviors
// - Must balance fleeing (expensive) vs. resting (risky)
// - Population grows steadily but is limited by predation
//
// Key asymmetries with Predators:
// - **Vision**: Prey have 360° vision (vs. predators' 60° cone)
//   Rationale: Prey are "prey" - they need to watch all directions for threats.
//   Predators can afford tunnel vision when hunting.
//
// - **Reproduction**: Prey reproduce on a timer (vs. predators' kill-threshold)
//   Rationale: Prey populations need to grow to sustain predators. Timer-based
//   reproduction ensures steady growth independent of energy (as long as they
//   survive long enough).
//
// - **Energy recovery**: Prey can rest to recover energy (predators cannot)
//   Rationale: Creates the "rest when safe, flee when threatened" dynamic.
// ============================================================================

/// A prey animal that tries to avoid predators.
#[derive(Clone)]
pub struct Prey {
    /// Core animal state (position, angle, energy, brain)
    pub core: AnimalCore,

    /// Timer tracking frames since birth/last reproduction.
    ///
    /// Design rationale: Prey reproduce based on time, not on "eating prey".
    /// This counter increments each frame and reproduction happens when it
    /// reaches a threshold (PREY_REPRODUCATION_RATE * FPS).
    ///
    /// This creates predictable, steady population growth (if prey survive),
    /// which is important for:
    /// 1. Preventing prey extinction (population recovers over time)
    /// 2. Providing a food source for predators
    /// 3. Creating evolutionary pressure (faster reproducers = more offspring)
    pub rest_time: i32,
}

/// Trait implementation for spatial hash queries (same as Predator).
impl HasPos for Prey {
    fn x(&self) -> f32 {
        self.core.x()
    }
    fn y(&self) -> f32 {
        self.core.y()
    }
    fn set_pos(&mut self, x: f32, y: f32) {
        self.core.set_xy(x, y);
    }
}

impl Prey {
    /// Creates a new prey with a random brain and position.
    ///
    /// Design rationale: Each prey starts with:
    /// 1. Random heading: Creates diverse initial movement patterns
    /// 2. Fresh neural network: Initialized with `prey_init_mut()` mutation rate
    /// 3. Full energy: Gives new prey a fair start
    /// 4. Zero rest_time: Must survive to reproduce
    pub fn new<R: Rng>(x: f32, y: f32, rng: &mut R) -> Self {
        let angle = rng.gen_range(0.0..TWO_PI);
        let brain = NeuralNetwork::new(NUMBER_SIGHTS_PREY, 2, prey_init_mut(), bias(), rng);

        Self {
            core: AnimalCore::new_with_brain(vec2(x, y), angle, PREY_ENERGY, brain),
            rest_time: 0,
        }
    }

    /// Returns the unique ID of this prey.
    #[inline]
    pub fn id(&self) -> usize {
        self.core.id
    }

    /// Computes sensory inputs for the prey's neural network.
    ///
    /// Prey have 360° vision divided into NUMBER_SIGHTS_PREY angular sectors.
    /// Each sector acts as a boolean "predator detector" (1.0 if predator detected,
    /// 0.0 otherwise).
    ///
    /// Design rationale:
    /// 1. **360° coverage**: Unlike predators with forward-facing vision, prey can
    ///    see in all directions simultaneously. This asymmetry is realistic:
    ///    - Predators hunt using focused vision
    ///    - Prey survive by detecting threats from any direction
    ///    - Creates interesting coevolutionary dynamics (prey develop omnidirectional
    ///      awareness, predators develop ambush tactics)
    ///
    /// 2. **Sector-based vision**: The 360° view is divided into equal sectors
    ///    centered on the prey's current heading. For example, with 8 sectors:
    ///    - Sector 0: [prey_angle - PI, prey_angle - PI + 45°)
    ///    - Sector 1: [prey_angle - PI + 45°, prey_angle - PI + 90°)
    ///    - etc.
    ///    
    ///    This is simpler than the predator's angular-width calculation because:
    ///    - We only care about "is there a predator in this general direction?"
    ///    - No need for precise angular size (prey just need to know where to flee)
    ///  
    /// 3. **Range limit**: Prey can't see infinitely far (SIGHT_RANGE_PREY).
    ///    This creates tactical depth:
    ///    - Prey might not see distant predators approaching
    ///    - Successful prey need to react quickly when predators enter range
    ///
    /// 4. **Body-relative sectors**: Sectors are defined relative to the prey's
    ///    current heading, not world coordinates. This means:
    ///    - Sector 0 is always "behind the prey"
    ///    - The brain learns direction-based responses ("flee forward if predator behind")
    ///    - Rotation-invariant behavior (works regardless of absolute heading)
    ///
    /// # Algorithm:
    /// 1. Calculate angle from prey to predator (in world space)
    /// 2. Convert to prey-relative angle (subtract prey's heading)
    /// 3. Map from [-π, π] to [0, 2π] for positive sector indexing
    /// 4. Divide by sector size to get sector index
    /// 5. Use rem_euclid to handle wraparound (sector n wraps to 0)
    ///
    /// # Returns
    /// Vector of floats (0.0 or 1.0) representing each vision sector
    pub fn sense_predators<'a, I>(&self, predators: I) -> Vec<f32>
    where
        I: IntoIterator<Item = &'a Predator>,
    {
        let n = NUMBER_SIGHTS_PREY.max(1);
        let mut inputs = vec![0.0; n];

        // Size of each angular sector (in radians)
        let sector_size = TWO_PI / (n as f32);
        let prey_pos = self.core.pos;
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        for pred in predators {
            let pred_pos = pred.core.pos;

            // Compute shortest vector to predator (toroidal world)
            let delta = wrapped_distance_vector(prey_pos, pred_pos, world_w, world_h);
            let dist = delta.length();

            // Skip if out of vision range
            if dist >= SIGHT_RANGE_PREY {
                continue;
            }

            // Calculate angle to predator in world coordinates
            let angle_to_pred = delta.y.atan2(delta.x);

            // Convert to prey-relative angle (which direction relative to prey's heading?)
            let rel = normalize_angle(angle_to_pred - self.core.angle); // [-PI, PI]

            // Map to [0, TWO_PI) for positive indexing
            let shifted = rel + PI;

            // Find which sector this predator falls into
            let idx = (shifted / sector_size).floor() as i32;
            let idx = idx.rem_euclid(n as i32) as usize; // Handle wraparound

            // Mark this sector as "predator detected"
            inputs[idx] = 1.0;
        }

        inputs
    }

    /// Executes one movement/decision step for the prey.
    ///
    /// Design rationale:
    /// 1. **No passive decay**: Unlike predators, prey don't lose energy just for
    ///    existing. Energy loss only happens through movement. This asymmetry:
    ///    - Allows prey to "hide" by staying still
    ///    - Creates a rest/flee decision dynamic
    ///    - Makes prey survival more about smart movement than constant feeding
    ///
    /// 2. **Rest mechanic**: If prey meets BOTH conditions:
    ///    - No danger detected (no input > 0.5)
    ///    - Energy below 60% of maximum
    ///    
    ///    Then the prey rests (gains PREY_REST_ENERGY_GAIN, does NOT move).
    ///    
    ///    Rationale: This creates emergent behavior:
    ///    - Prey learn to rest when safe to recover energy
    ///    - Prey must flee when threatened, even if low on energy
    ///    - Creates spatial patterns (prey accumulate in "safe zones")
    ///    - Evolutionary pressure for energy-efficient evasion
    ///
    /// 3. **Danger detection**: We check if ANY vision sector has value > 0.5.
    ///    Since vision outputs are binary (0.0 or 1.0), this effectively means
    ///    "is any predator visible?" This prevents resting during danger.
    ///
    /// 4. **Negative energy handling**: If energy goes negative (from expensive
    ///    fleeing), we force a minimum speed of 0.1. This prevents prey from
    ///    "giving up" and stopping, creating a last-ditch escape attempt.
    ///    
    ///    Rationale: Negative energy represents exhaustion, but animals don't
    ///    just stop moving - they keep trying to escape even when exhausted.
    ///    This makes the simulation more dynamic.
    ///
    /// 5. **Commented-out code**: Shows alternative mechanics that were considered:
    ///    - Forcing minimum speed during danger: would make fleeing mandatory
    ///    - Different rest threshold: could change when prey choose to rest
    ///    
    ///    These were left as comments to document the design exploration.
    ///
    /// # Arguments
    /// * `inputs` - Sensory inputs from sense_predators (vision sectors)
    pub fn move_step(&mut self, inputs: &[f32]) {
        let energy_ratio = self.core.energy / PREY_ENERGY;
        let outputs = self.core.brain.forward_vectorized(inputs, energy_ratio);

        let speed_factor = outputs[0].clamp(0.0, 1.0);

        // Check if any predators are visible (any sensor > 0.5)
        let danger = inputs.iter().any(|&v| v > 0.5);

        // Rest mechanic: recover energy when safe and low on energy
        // This creates the "flee when threatened, rest when safe" behavior
        if !danger && self.core.energy < 0.6 * PREY_ENERGY {
            self.core.energy = (self.core.energy + PREY_REST_ENERGY_GAIN).min(PREY_ENERGY);
            return; // Don't move while resting
        }

        // Alternative rest conditions that were considered:
        // if !danger && speed_factor < 0.6 * PREY_ENERGY {  // Rest based on brain output

        let mut speed_factor = speed_factor;

        // Possible enhancement: force minimum escape speed during danger
        // let min_escape = if danger { 0.25 } else { 0.0 };
        // speed_factor = speed_factor.max(min_escape);

        // Exhaustion handling: even with negative energy, keep moving slightly
        // This simulates an exhausted animal making last-ditch escape attempts
        if self.core.energy < 0.0 {
            speed_factor = speed_factor.max(0.1);
        }

        // Apply turning and movement
        let turn_delta = outputs[1].clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_2;

        move_with_speed_factor(
            &mut self.core,
            speed_factor,
            turn_delta,
            PREY_SPEED,
            PREY_MOVING_DECAY,
            PREY_ENERGY,
        );
    }

    /// Attempts to reproduce, creating an offspring if conditions are met.
    ///
    /// Design rationale:
    /// 1. **Timer-based reproduction**: Unlike predators (kill-based), prey reproduce
    ///    after surviving for a certain time. The timer (rest_time) increments every
    ///    frame, and reproduction happens when it reaches:
    ///    `PREY_REPRODUCATION_RATE * FRAMES_PER_SECOND`
    ///    
    ///    For example, if PREY_REPRODUCATION_RATE = 20.0 and FPS = 60:
    ///    - Threshold = 20 * 60 = 1200 frames
    ///    - At 60 FPS, this is 20 seconds of survival
    ///    
    ///    Rationale:
    ///    - Prey don't have an "eat prey" mechanic, so time-based makes sense
    ///    - Rewards survival skill directly
    ///    - Creates steady population growth (if prey survive)
    ///
    /// 2. **Population control via `has_slot`**: The caller (game logic) can limit
    ///    total prey population by controlling this parameter. If `has_slot` is false,
    ///    the prey is "ready" to reproduce but can't until a slot opens.
    ///    
    ///    Crucially: We DON'T reset `rest_time` when slot is unavailable. Instead,
    ///    we clamp it at the threshold. This means:
    ///    - Prey "remembers" being ready to reproduce
    ///    - As soon as a slot opens, this prey can immediately reproduce
    ///    - Creates a "queue" of ready-to-reproduce prey
    ///    
    ///    Rationale: Prevents prey from "wasting" their readiness when populations
    ///    are at capacity. Without this, prey would have to wait another full cycle
    ///    after slots open up.
    ///
    /// 3. **Larger spawn offset (±50 pixels)**: Much larger than predators' ±1 pixel.
    ///    
    ///    Rationale:
    ///    - Prey reproduce more often than predators
    ///    - Larger offset prevents dense clustering
    ///    - Spreads prey population spatially
    ///    - Reduces local resource competition (though there are no explicit resources)
    ///    - Makes population distributions more visually natural
    ///
    /// 4. **Brain inheritance**: Same as predators - child gets mutated copy of
    ///    parent's brain, enabling evolution of better survival strategies.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `has_slot` - Whether the population has room for a new prey
    ///
    /// # Returns
    /// `Some(Prey)` if reproduction successful, `None` otherwise
    pub fn reproduce<R: Rng>(&mut self, rng: &mut R, has_slot: bool) -> Option<Prey> {
        // Increment timer every frame
        self.rest_time += 1;

        // Calculate reproduction threshold (time in frames)
        let threshold = (PREY_REPRODUCATION_RATE * FRAMES_PER_SECOND as f32) as i32;

        // Not ready yet
        if self.rest_time < threshold {
            return None;
        }

        // Ready to reproduce, but population is at capacity
        // (German: "Sie wäre jetzt bereit. Aber kein Slot frei ist:")
        if !has_slot {
            // Clamp at threshold rather than resetting
            // (German: "nicht zurücksetzen, sonst verliert sie den 'ready'-Status")
            self.rest_time = threshold;
            return None;
        }

        // Slot is free -> give birth!
        // (German: "Slot ist frei -> Geburt")
        self.rest_time = 0; // Reset timer for next reproduction

        // Spawn child with larger offset than predators (±50 vs ±1)
        let ox =
            rng.gen_range((self.core.pos.x as i32 - 50)..=(self.core.pos.x as i32 + 50)) as f32;
        let oy =
            rng.gen_range((self.core.pos.y as i32 - 50)..=(self.core.pos.y as i32 + 50)) as f32;

        // Create child with inherited, mutated brain
        let mut child = Prey::new(ox, oy, rng);
        child.core.brain = inherited_brain_with_mutations(&self.core.brain, rng);

        Some(child)
    }

    /// Draws the prey's 360° vision rays for debugging/visualization.
    ///
    /// Design rationale: Visualizes the omnidirectional vision. Unlike predators
    /// (cone), prey vision rays form a complete circle around them.
    pub fn draw_sight(&self) {
        let n = NUMBER_SIGHTS_PREY.max(1);
        let step = TWO_PI / (n as f32); // Angular spacing between rays
        let world_w = SCREEN_WIDTH as f32;
        let world_h = SCREEN_HEIGHT as f32;

        // Draw each vision ray evenly spaced around the circle
        for i in 0..n {
            let sight_angle = self.core.angle + step * (i as f32);

            let end_x = self.core.pos.x + SIGHT_RANGE_PREY * sight_angle.cos();
            let end_y = self.core.pos.y + SIGHT_RANGE_PREY * sight_angle.sin();

            draw_wrapped_line(
                self.core.pos,
                vec2(end_x, end_y),
                world_w,
                world_h,
                1.0,
                SKYBLUE, // Sky blue distinguishes prey vision from predator vision (yellow)
            );
        }
    }

    /// Draws the prey as a green circle with vision rays.
    ///
    /// Design rationale: Green color clearly distinguishes prey from predators
    /// (red), creating immediate visual feedback on population dynamics.
    pub fn draw(&self) {
        draw_circle(self.core.pos.x, self.core.pos.y, PREY_RADIUS, GREEN);
        self.draw_sight();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORLD_W: f32 = 1000.0;
    const WORLD_H: f32 = 1000.0;
    const EPSILON: f32 = 0.0001;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec_approx_eq(a: Vec2, b: Vec2) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y)
    }

    // ==================== wrap_position tests ====================

    #[test]
    fn test_wrap_position_no_wrap_needed() {
        let pos = vec2(500.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, pos));
    }

    #[test]
    fn test_wrap_position_x_overflow() {
        let pos = vec2(1050.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(50.0, 500.0)));
    }

    #[test]
    fn test_wrap_position_x_underflow() {
        let pos = vec2(-50.0, 500.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(950.0, 500.0)));
    }

    #[test]
    fn test_wrap_position_y_overflow() {
        let pos = vec2(500.0, 1100.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(500.0, 100.0)));
    }

    #[test]
    fn test_wrap_position_y_underflow() {
        let pos = vec2(500.0, -100.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(500.0, 900.0)));
    }

    #[test]
    fn test_wrap_position_both_overflow() {
        let pos = vec2(1200.0, 1300.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(200.0, 300.0)));
    }

    #[test]
    fn test_wrap_position_corner_underflow() {
        let pos = vec2(-200.0, -150.0);
        let wrapped = wrap_position(pos, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(wrapped, vec2(800.0, 850.0)));
    }

    // ==================== wrapped_distance_vector tests ====================

    #[test]
    fn test_wrapped_delta_no_wrap_needed() {
        let a = vec2(400.0, 400.0);
        let b = vec2(600.0, 600.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(200.0, 200.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_x_positive() {
        // a is at x=900, b is at x=100
        // Direct distance: 100 - 900 = -800
        // Wrapped distance: should go right across border = 200
        let a = vec2(900.0, 500.0);
        let b = vec2(100.0, 500.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(200.0, 0.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_x_negative() {
        // a is at x=100, b is at x=900
        // Direct distance: 900 - 100 = 800
        // Wrapped distance: should go left across border = -200
        let a = vec2(100.0, 500.0);
        let b = vec2(900.0, 500.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(-200.0, 0.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_y_positive() {
        let a = vec2(500.0, 950.0);
        let b = vec2(500.0, 50.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(0.0, 100.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_y_negative() {
        let a = vec2(500.0, 50.0);
        let b = vec2(500.0, 950.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(0.0, -100.0)));
    }

    #[test]
    fn test_wrapped_delta_wrap_corner() {
        // a at corner (950, 950), b at corner (50, 50)
        // Should wrap both x and y to get shortest path
        let a = vec2(950.0, 950.0);
        let b = vec2(50.0, 50.0);
        let delta = wrapped_distance_vector(a, b, WORLD_W, WORLD_H);
        assert!(vec_approx_eq(delta, vec2(100.0, 100.0)));
    }

    // ==================== wrapped_distance_abs tests ====================

    #[test]
    fn test_wrapped_distance_same_point() {
        let a = vec2(500.0, 500.0);
        let dist = wrapped_distance_abs(a, a, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 0.0));
    }

    #[test]
    fn test_wrapped_distance_no_wrap() {
        let a = vec2(0.0, 0.0);
        let b = vec2(100.0, 0.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 100.0));
    }

    #[test]
    fn test_wrapped_distance_across_x_border() {
        // Points near opposite edges should have small wrapped distance
        let a = vec2(10.0, 500.0);
        let b = vec2(990.0, 500.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 20.0));
    }

    #[test]
    fn test_wrapped_distance_across_y_border() {
        let a = vec2(500.0, 5.0);
        let b = vec2(500.0, 995.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 10.0));
    }

    #[test]
    fn test_wrapped_distance_across_corner() {
        // Points at opposite corners, close via wrapping
        let a = vec2(10.0, 10.0);
        let b = vec2(990.0, 990.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        // dx = 20, dy = 20, distance = sqrt(800) ≈ 28.28
        let expected = (20.0_f32.powi(2) + 20.0_f32.powi(2)).sqrt();
        assert!(approx_eq(dist, expected));
    }

    #[test]
    fn test_wrapped_distance_half_world() {
        // At exactly half the world width, both paths are equal
        let a = vec2(0.0, 500.0);
        let b = vec2(500.0, 500.0);
        let dist = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        assert!(approx_eq(dist, 500.0));
    }

    #[test]
    fn test_wrapped_distance_symmetry() {
        let a = vec2(100.0, 200.0);
        let b = vec2(900.0, 800.0);
        let dist_ab = wrapped_distance_abs(a, b, WORLD_W, WORLD_H);
        let dist_ba = wrapped_distance_abs(b, a, WORLD_W, WORLD_H);
        assert!(approx_eq(dist_ab, dist_ba));
    }

    // ==================== normalize_angle tests ====================

    #[test]
    fn test_normalize_angle_zero() {
        assert!(approx_eq(normalize_angle(0.0), 0.0));
    }

    #[test]
    fn test_normalize_angle_positive_within_range() {
        let angle = std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), angle));
    }

    #[test]
    fn test_normalize_angle_negative_within_range() {
        let angle = -std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), angle));
    }

    #[test]
    fn test_normalize_angle_over_pi() {
        // 3*PI/2 should wrap to -PI/2
        let angle = 3.0 * std::f32::consts::FRAC_PI_2;
        let expected = -std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), expected));
    }

    #[test]
    fn test_normalize_angle_under_minus_pi() {
        // -3*PI/2 should wrap to PI/2
        let angle = -3.0 * std::f32::consts::FRAC_PI_2;
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(approx_eq(normalize_angle(angle), expected));
    }

    #[test]
    fn test_normalize_angle_two_pi() {
        // 2*PI should wrap to 0
        let angle = 2.0 * std::f32::consts::PI;
        assert!(approx_eq(normalize_angle(angle), 0.0));
    }
}
