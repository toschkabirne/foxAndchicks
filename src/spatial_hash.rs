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
    /// Size of a cell in pixels
    pub cell_size: i32,
    /// Storage for indices into a Vec<T>
    pub cells: HashMap<(i32, i32), Vec<usize>>,
    /// Number of cells in x direction (needed for toroidal wrapping)
    pub num_cells_x: i32,
    /// Number of cells in y direction (needed for toroidal wrapping)
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
        // Wrap cell coordinates for toroidal world
        (
            cx.rem_euclid(self.num_cells_x),
            cy.rem_euclid(self.num_cells_y),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPoint {
        x: f32,
        y: f32,
    }

    impl HasPos for TestPoint {
        fn x(&self) -> f32 {
            self.x
        }
        fn y(&self) -> f32 {
            self.y
        }
        fn set_pos(&mut self, _x: f32, _y: f32) {}
    }

    /// Tests that an object inside the query cell is detected.
    ///
    /// ```text
    /// World Width: 1000, Cell Size: 200
    /// Grid: [0, 200) [200, 400) [400, 600) ...
    /// Index:    0         1         2
    ///
    /// Query (500,500) -> Cell 2
    /// Object (799,500) -> Cell 3 (Neighbor of 2)
    /// Result: FOUND
    /// ```
    #[test]
    fn test_inclusion_x() {
        let cell_size = 200;
        let mut sh = SpatialHash::new(cell_size, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 799.0, y: 500.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 500.0);
        assert!(
            !nearby.is_empty(),
            "Object within neighbor range should be found"
        );
        assert_eq!(nearby[0], 0);
    }

    /// Tests that an object just outside the neighbor range is excluded.
    ///
    /// ```text
    /// Query at Cell 2.
    /// Neighbors: 1, 2, 3.
    ///
    ///  [Cell 1]  [Cell 2]  [Cell 3]  ||  [Cell 4]
    ///               ^ Query              ^ Object (800.0)
    ///  <----- Neighborhood ------>   ||  (Too far)
    /// ```
    #[test]
    fn test_exclusion_x() {
        let cell_size = 200;
        let mut sh = SpatialHash::new(cell_size, 1000.0, 1000.0);

        // 800.0 falls exactly into Cell 4 (800 / 200 = 4).
        // Cell 4 is NOT a neighbor of Cell 2.
        let objs = vec![TestPoint { x: 800.0, y: 500.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 500.0);
        assert!(
            nearby.is_empty(),
            "Object in Cell 4 should not be seen by Cell 2"
        );
    }

    /// Tests Y-axis neighbor detection.
    ///
    /// ```text
    /// Y-Axis Grid:
    /// Cell 2 (400-600)  <- Query (500)
    /// Cell 1 (200-400)  <- Object (200)
    ///
    /// Cell 1 is a direct neighbor of Cell 2.
    /// ```
    #[test]
    fn test_inclusion_y() {
        let cell_size = 200;
        let mut sh = SpatialHash::new(cell_size, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 400.0, y: 200.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 500.0);
        assert!(!nearby.is_empty(), "Object in Y-neighbor should be found");
    }

    /// Tests Y-axis exclusion.
    ///
    /// ```text
    /// Y-Axis Grid:
    /// Cell 2 (400-600) <- Query (500)
    /// Cell 1 (200-400)
    /// Cell 0 (0-200)   <- Object (199)
    ///
    /// Cell 0 is NOT a neighbor of Cell 2 (Neighbors are 1, 2, 3).
    /// ```
    #[test]
    fn test_exclusion_y() {
        let cell_size = 200;
        let mut sh = SpatialHash::new(cell_size, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 400.0, y: 199.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 500.0);
        assert!(nearby.is_empty(), "Object in Cell 0 is too far from Cell 2");
    }

    /// Tests Toroidal Wrapping on the X-axis.
    /// The world acts like a cylinder/torus. The last cell is a neighbor of the first.
    ///
    /// ```text
    /// Cells: [0] [1] [2] [3] [4]
    ///         ^               ^
    ///      Query(5.0)      Object(800+)
    ///      
    /// Logical Connection:  ... [4] <-> [0] ...
    /// ```
    #[test]
    fn test_spatial_hash_wrapping_x() {
        let mut sh = SpatialHash::new(200, 1000.0, 1000.0);
        // Object at 800.0 is in Cell 4. Query at 5.0 is in Cell 0.
        // Cell 4 is the "left" neighbor of Cell 0 in a wrapping world.
        let objs = vec![TestPoint { x: 800.0, y: 500.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(5.0, 500.0);
        assert!(
            !nearby.is_empty(),
            "Toroidal neighbor (left-wrap) NOT detected"
        );
    }

    /// Tests Toroidal Wrapping on the Y-axis.
    ///
    /// ```text
    ///      [Cell 0] (y=5.0)  <- Query
    ///         ^
    ///         | (Wraps)
    ///         v
    ///      [Cell 4] (y=900+) <- Object
    /// ```
    #[test]
    fn test_spatial_hash_wrapping_y() {
        let mut sh = SpatialHash::new(200, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 500.0, y: 900.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 5.0);
        assert!(
            !nearby.is_empty(),
            "Toroidal neighbor (top-bottom wrap) NOT detected"
        );
    }

    /// Tests that wrapping works diagonally across corners.
    ///
    /// ```text
    /// +-----------+
    /// | Q . . . . |  Q = Query (Top-Left)
    /// | . . . . . |
    /// | . . . . O |  O = Object (Bottom-Right)
    /// +-----------+
    ///
    /// In a torus, Q and O are direct diagonal neighbors.
    /// ```
    #[test]
    fn test_diagonal_corner_wrap() {
        let mut sh = SpatialHash::new(200, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 995.0, y: 995.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(5.0, 5.0);
        assert!(
            !nearby.is_empty(),
            "Diagonal toroidal neighbor NOT detected"
        );
    }

    /// Tests specific floating point boundary condition.
    ///
    /// ```text
    /// World Width: 1000.0
    /// Object at 0.0 (Left edge)
    /// Query at 999.9 (Right edge)
    ///
    /// [0.0] <-----------------> [999.9]
    /// Cell 0                     Cell 4
    /// ```
    #[test]
    fn test_query_at_exact_world_boundary() {
        let mut sh = SpatialHash::new(200, 1000.0, 1000.0);
        let objs = vec![TestPoint { x: 0.0, y: 0.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(999.9, 999.9);
        assert!(
            !nearby.is_empty(),
            "Object at 0.0 should be found by query at world edge"
        );
    }

    /// Tests that the query retrieves objects from all 8 surrounding cells + center.
    ///
    /// ```text
    /// [X] [X] [X]
    /// [X] [Q] [X]  -> Q = Query, X = Object
    /// [X] [X] [X]
    /// ```
    #[test]
    fn test_neighborhood_density() {
        let mut sh = SpatialHash::new(200, 1000.0, 1000.0);
        let objs = vec![
            TestPoint { x: 500.0, y: 500.0 }, // Center
            TestPoint { x: 300.0, y: 300.0 }, // NW
            TestPoint { x: 700.0, y: 700.0 }, // SE
            TestPoint { x: 300.0, y: 700.0 }, // SW
            TestPoint { x: 700.0, y: 300.0 }, // NE
        ];
        sh.rebuild_from(&objs);

        let nearby = sh.query(500.0, 500.0);
        assert_eq!(
            nearby.len(),
            5,
            "Should find all 5 objects in the 3x3 neighborhood"
        );
    }

    /// Tests the 1x1 world edge case.
    ///
    /// ```text
    /// World Size == Cell Size
    /// +-------+
    /// | [0,0] |  -> Everything is a neighbor of everything.
    /// +-------+
    /// ```
    #[test]
    fn test_1x1_world() {
        let mut sh = SpatialHash::new(100, 100.0, 100.0);
        let objs = vec![TestPoint { x: 90.0, y: 90.0 }];
        sh.rebuild_from(&objs);

        let nearby = sh.query(10.0, 10.0);
        assert!(
            !nearby.is_empty(),
            "In a 1x1 world, all objects should be neighbors"
        );
    }

    /// Tests robustness against coordinates exactly matching world bounds.
    ///
    /// ```text
    /// Grid: [0...10)
    /// X = 1000.0 (Exactly World Width)
    ///
    /// floor(1000.0 / 100) = 10. Index 10 is technically OOB (0..9).
    /// rem_euclid(10, 10) = 0.
    /// Result: Should wrap safely to Cell 0.
    /// ```
    #[test]
    fn test_boundary_max_width() {
        let mut sh = SpatialHash::new(100, 1000.0, 1000.0);

        let objs = vec![TestPoint {
            x: 1000.0,
            y: 500.0,
        }];
        sh.rebuild_from(&objs);

        // Query at 0.0 (should find the object wrapped to cell 0)
        let nearby = sh.query(0.0, 500.0);
        assert!(
            !nearby.is_empty(),
            "Object at x=WorldWidth should wrap to 0"
        );
    }
}
