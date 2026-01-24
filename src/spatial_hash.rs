use std::collections::HashMap;

pub trait HasPos {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn set_pos(&mut self, x: f32, y: f32);
}

/// Design rationale: The spatial hash accelerates "find nearby entities" queries
/// from O(n) to O(1) average case. This is critical for performance when checking:
#[derive(Clone)]
pub struct SpatialHash {
    pub cell_size: i32,
    pub cells: HashMap<(i32, i32), Vec<usize>>, // store indices into a Vec<T>
    /// Number of cells in x direction (for toroidal wrapping)
    pub num_cells_x: i32,
    /// Number of cells in y direction (for toroidal wrapping)
    pub num_cells_y: i32,
}

impl SpatialHash {
    pub fn new(cell_size: i32, world_width: f32, world_height: f32) -> Self {
        let cell_size = cell_size.max(1);
        Self {
            cell_size,
            cells: HashMap::new(),
            num_cells_x: (world_width / cell_size as f32).ceil() as i32,
            num_cells_y: (world_height / cell_size as f32).ceil() as i32,
        }
    }

    #[inline]
    fn hash(&self, x: f32, y: f32) -> (i32, i32) {
        let cx = (x / self.cell_size as f32).floor() as i32;
        let cy = (y / self.cell_size as f32).floor() as i32;
        (cx, cy)
    }

    /// Insert an object index at a given position (position is passed in so hash doesn't need the slice).
    #[inline]
    pub fn insert_at(&mut self, idx: usize, x: f32, y: f32) {
        let cell = self.hash(x, y);
        self.cells.entry(cell).or_default().push(idx);
    }

    /// Rebuild helper: clear + insert all indices from a slice.
    pub fn rebuild_from<T: HasPos>(&mut self, objs: &[T]) {
        self.clear();
        for (i, o) in objs.iter().enumerate() {
            self.insert_at(i, o.x(), o.y());
        }
    }

    /// Query indices of objects in 3x3 neighbor cells around (x,y).
    pub fn query(&self, x: f32, y: f32) -> Vec<usize> {
        let mut out = Vec::new();
        self.query_into(&mut out, x, y);
        out
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

/// Fill `out` with indices in the 3x3 neighbor cells around (x,y).
/// Reuses capacity of `out` to avoid allocations in the hot loop.
/// Uses toroidal wrapping so cells at world edges can see across borders.
impl SpatialHash {
    pub fn query_into(&self, out: &mut Vec<usize>, x: f32, y: f32) {
        out.clear();
        let cell = self.hash(x, y);

        for dx in [-1, 0, 1] {
            for dy in [-1, 0, 1] {
                // Wrap cell coordinates for toroidal world
                let wrapped_x = (cell.0 + dx).rem_euclid(self.num_cells_x);
                let wrapped_y = (cell.1 + dy).rem_euclid(self.num_cells_y);
                let neighbor_cell = (wrapped_x, wrapped_y);
                if let Some(list) = self.cells.get(&neighbor_cell) {
                    out.extend(list.iter().copied());
                }
            }
        }
    }
}
