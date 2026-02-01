#pragma once
#include "engine/engine.hpp"
#include "engine/common/utils.hpp"
#include "engine/common/index_vector.hpp"

#include "user/configuration.hpp"
#include "user/simulation/simulation_configuration.hpp"

#include "collision_grid.hpp"
#include "physic_object.hpp"
#include "position_constraint.hpp"


struct PhysicSolver : public pez::core::IProcessor
{
    SIVector<PhysicObject>       objects;
    SIVector<PositionConstraint> constraints;
    CollisionGrid                grid;
    Vec2                         world_size;

    // Simulation solving pass count
    uint32_t        sub_steps;
    tp::ThreadPool& thread_pool;

    explicit
    PhysicSolver(SimulationConfiguration const& conf)
        : grid{conf.world_size}
        , world_size{conf.world_size_f}
        , sub_steps{1}
        , thread_pool{pez::core::getSingleton<tp::ThreadPool>()}
    {
        grid.clear();
    }

    // Checks if two atoms are colliding and if so create a new contact
    void solveContact(uint32_t atom_1_idx, uint32_t atom_2_idx)
    {
        constexpr float response_coef = 0.8f;
        constexpr float eps           = 0.0001f;
        PhysicObject& obj_1 = objects.getData()[atom_1_idx];
        PhysicObject& obj_2 = objects.getData()[atom_2_idx];
        const Vec2 o2_o1  = obj_1.position - obj_2.position;
        const float dist2 = o2_o1.x * o2_o1.x + o2_o1.y * o2_o1.y;
        if (dist2 < 1.0f && dist2 > eps) {
            const float dist = std::sqrt(dist2);
            // Radius are all equal to 1.0f
            const float delta  = response_coef * (1.0f - dist);
            const float mass_sum = obj_1.mass + obj_2.mass;
            const Vec2 col_vec = (o2_o1 / dist) * delta;
            obj_1.position += col_vec * (obj_2.mass / mass_sum);
            obj_2.position -= col_vec * (obj_1.mass / mass_sum);
        }
    }

    [[nodiscard]]
    Vec2 getObjectPosition(siv::ID object_id) const
    {
        return objects[object_id].position;
    }

    void checkAtomCellCollisions(uint32_t atom_idx, const CollisionCell& c)
    {
        for (uint32_t i{0}; i < c.objects_count; ++i) {
            solveContact(atom_idx, c.objects[i]);
        }
    }

    void processCell(const CollisionCell& c, uint32_t index)
    {
        for (uint32_t i{0}; i < c.objects_count; ++i) {
            const uint32_t atom_idx = c.objects[i];
            checkAtomCellCollisions(atom_idx, grid.data[index - 1]);
            checkAtomCellCollisions(atom_idx, grid.data[index]);
            checkAtomCellCollisions(atom_idx, grid.data[index + 1]);
            checkAtomCellCollisions(atom_idx, grid.data[index + grid.height - 1]);
            checkAtomCellCollisions(atom_idx, grid.data[index + grid.height    ]);
            checkAtomCellCollisions(atom_idx, grid.data[index + grid.height + 1]);
            checkAtomCellCollisions(atom_idx, grid.data[index - grid.height - 1]);
            checkAtomCellCollisions(atom_idx, grid.data[index - grid.height    ]);
            checkAtomCellCollisions(atom_idx, grid.data[index - grid.height + 1]);
        }
    }

    void solveCollisionThreaded(uint32_t start, uint32_t end)
    {
        for (uint32_t idx{start}; idx < end; ++idx) {
            processCell(grid.data[idx], idx);
        }
    }

    // Find colliding atoms
    void solveCollisions()
    {
        // Multi-thread grid
        const uint32_t thread_count = thread_pool.m_thread_count;
        const uint32_t slice_count  = thread_count * 2;
        const uint32_t slice_size   = (grid.width / slice_count) * grid.height;
        const uint32_t last_cell    = (2 * (thread_count - 1) + 2) * slice_size;
        // Find collisions in two passes to avoid data races

        // First collision pass
        for (uint32_t i{0}; i < thread_count; ++i) {
            thread_pool.addTask([this, i, slice_size]{
                uint32_t const start{2 * i * slice_size};
                uint32_t const end  {start + slice_size};
                solveCollisionThreaded(start, end);
            });
        }
        // Eventually process rest if the world is not divisible by the thread count
        if (last_cell < grid.data.size()) {
            thread_pool.addTask([this, last_cell]{
                solveCollisionThreaded(last_cell, to<uint32_t>(grid.data.size()));
            });
        }
        thread_pool.waitForCompletion();
        // Second collision pass
        for (uint32_t i{0}; i < thread_count; ++i) {
            thread_pool.addTask([this, i, slice_size]{
                uint32_t const start{(2 * i + 1) * slice_size};
                uint32_t const end  {start + slice_size};
                solveCollisionThreaded(start, end);
            });
        }
        thread_pool.waitForCompletion();
    }

    void solveConstraints()
    {
        auto const& raw_data = constraints.getData();
        thread_pool.dispatch(static_cast<uint32_t>(raw_data.size()), [this, &raw_data](uint32_t start, uint32_t end) {
            for (uint32_t i{start}; i < end; ++i) {
                PositionConstraint const& constraint = raw_data[i];
                constraint.apply(objects[constraint.object_id]);
            }
        });
    }

    siv::ID createConstraint(siv::ID object_id, Vec2 target, float strength)
    {
        siv::ID const id = constraints.emplace_back(object_id, target, strength);
        return id;
    }

    void removeConstraint(siv::ID constraint_id)
    {
        constraints.erase(constraint_id);
    }

    // Add a new object to the solver
    pez::core::ID addObject(const PhysicObject& object)
    {
        return to<pez::core::ID>(objects.push_back(object));
    }

    // Add a new object to the solver
    pez::core::ID createObject(Vec2 pos, pez::core::ID agent_id)
    {
        return to<pez::core::ID>(objects.emplace_back(pos, agent_id));
    }

    void removeObject(pez::core::ID object_id)
    {
        objects.erase(object_id);
    }

    [[nodiscard]]
    PhysicObject& getObject(pez::core::ID object_id)
    {
        return objects[object_id];
    }

    [[nodiscard]]
    PhysicObject const& getObject(pez::core::ID object_id) const
    {
        return objects[object_id];
    }

    [[nodiscard]]
    PhysicObject const& getObjectRaw(uint32_t idx) const
    {
        return objects.getData()[idx];
    }

    void update(float dt) override
    {
        // Perform the sub steps
        const float sub_dt = dt / static_cast<float>(sub_steps);
        for (uint32_t i(sub_steps); i--;) {
            addObjectsToGrid();
            solveCollisions();
            solveConstraints();
            updateObjects_multi(sub_dt);
        }
    }

    void addObjectsToGrid()
    {
        grid.clear();
        // Safety border to avoid adding object outside the grid
        uint32_t i{0};
        auto const& objects_data{objects.getData()};
        for (const PhysicObject& obj : objects_data) {
            if (obj.position.x > 1.0f && obj.position.x < world_size.x - 1.0f &&
                obj.position.y > 1.0f && obj.position.y < world_size.y - 1.0f) {
                grid.addAtom(to<int32_t>(obj.position.x), to<int32_t>(obj.position.y), i);
            }
            ++i;
        }
    }

    void clear()
    {
        grid.clear();
        objects.clear();
        constraints.clear();
    }

    void updateObjects_multi(float dt)
    {
        thread_pool.dispatch(to<uint32_t>(objects.size()), [&](uint32_t start, uint32_t end){
            for (uint32_t i{start}; i < end; ++i) {
                PhysicObject& obj = objects.getData()[i];
                // Apply Verlet integration
                obj.update(dt);
                // Apply map borders collisions
                const float margin = 2.0f;
                if (obj.position.x > world_size.x - margin) {
                    obj.position.x = world_size.x - margin;
                } else if (obj.position.x < margin) {
                    obj.position.x = margin;
                }
                if (obj.position.y > world_size.y - margin) {
                    obj.position.y = world_size.y - margin;
                } else if (obj.position.y < margin) {
                    obj.position.y = margin;
                }
            }
        });
    }

    template<typename TCallback>
    void foreachCollider(Vec2 position, TCallback&& callback) const
    {
        auto const& data = objects.getData();
        const IVec2 grid_coord{to<int32_t>(position.x), to<int32_t>(position.y)};
        for (int32_t y{grid_coord.y - 1}; y <= grid_coord.y + 1; ++y) {
            for (int32_t x{grid_coord.x - 1}; x <= grid_coord.x + 1; ++x) {
                const CollisionCell& cell = grid.get(x, y);
                for (uint32_t i{0}; i<cell.objects_count; ++i) {
                    const PhysicObject& object = data[cell.objects[i]];
                    const Vec2          v      = object.position - position;
                    const float         dist2  = v.x * v.x + v.y * v.y;
                    if (dist2 < 1.0f) {
                        if (callback(object, v / std::sqrt(dist2))) { return; }
                    }
                }
            }
        }
    }
};
