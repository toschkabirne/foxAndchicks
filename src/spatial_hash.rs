// src/spatial_hash.rs

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub trait HasPos {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn set_pos(&mut self, x: f32, y: f32);
}

#[derive(Clone)]
pub struct SpatialHash<T: HasPos> {
    pub cell_size: i32,
    pub cells: HashMap<(i32, i32), Vec<Rc<RefCell<T>>>>,
}

impl<T: HasPos> SpatialHash<T> {
    pub fn new(cell_size: i32) -> Self {
        Self {
            cell_size: cell_size.max(1),
            cells: HashMap::new(),
        }
    }

    fn hash(&self, x: f32, y: f32) -> (i32, i32) {
        let cx = (x / self.cell_size as f32).floor() as i32;
        let cy = (y / self.cell_size as f32).floor() as i32;
        (cx, cy)
    }

    pub fn insert(&mut self, obj: Rc<RefCell<T>>) {
        let (x, y) = {
            let o = obj.borrow();
            (o.x(), o.y())
        };
        let cell = self.hash(x, y);
        self.cells.entry(cell).or_default().push(obj);
    }

    // Python: move(self, obj, new_x, new_y)
    pub fn move_obj(&mut self, obj: &Rc<RefCell<T>>, new_x: f32, new_y: f32) {
        let (old_x, old_y) = {
            let o = obj.borrow();
            (o.x(), o.y())
        };

        let old_cell = self.hash(old_x, old_y);
        let new_cell = self.hash(new_x, new_y);

        if old_cell != new_cell {
            if let Some(list) = self.cells.get_mut(&old_cell) {
                // remove(obj) by identity, like Python list.remove(obj)
                if let Some(pos) = list.iter().position(|rc| Rc::ptr_eq(rc, obj)) {
                    list.remove(pos);
                }
                if list.is_empty() {
                    self.cells.remove(&old_cell);
                }
            }

            {
                let mut o = obj.borrow_mut();
                o.set_pos(new_x, new_y);
            }
            self.insert(Rc::clone(obj));
        } else {
            let mut o = obj.borrow_mut();
            o.set_pos(new_x, new_y);
        }
    }

    pub fn query(&self, x: f32, y: f32) -> Vec<Rc<RefCell<T>>> {
        let cell = self.hash(x, y);
        let mut neighbors: Vec<Rc<RefCell<T>>> = Vec::new();

        for dx in [-1, 0, 1] {
            for dy in [-1, 0, 1] {
                let neighbor_cell = (cell.0 + dx, cell.1 + dy);
                if let Some(list) = self.cells.get(&neighbor_cell) {
                    neighbors.extend(list.iter().cloned());
                }
            }
        }

        neighbors
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }
}
