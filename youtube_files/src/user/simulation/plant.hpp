#pragma once
#include "engine/engine.hpp"
#include "engine/common/grid.hpp"


/** Represents a cell of the plant grid
 *
 */
struct GroundCell
{
    float fertilize_boost = 0.0f;
    bool  occupied         = false;
};


/** Representation of a Plant entity
 *
 */
struct Plant : public pez::core::Entity
{
    explicit
    Plant(pez::core::EntityID id_, Vec2 position_, float init_reserve)
        : pez::core::Entity{id_}
        , position{position_}
        , position_top{position_}
        , reserve{init_reserve}
    {
        createPhysicsObject();

        // Avoid using RNG because it would screw existing simulations
        {
            float const seed = static_cast<float>(id.instance_id) * 2048.0f;
            float const angle = seed;
            float const scale = 1.5f + 0.25f * std::sin(seed);
            direction = scale * Vec2{std::cos(angle), std::sin(angle)};
        }

        {
            float const seed = static_cast<float>(id.instance_id) * 911.0f;
            float const angle = seed;
            float const scale = 1.5f + 0.25f * std::sin(seed);
            direction_top = 0.6f * scale * Vec2{std::cos(angle), std::sin(angle)};
        }
    }

    void onRemove() override
    {
        auto& solver{pez::core::getProcessor<PhysicSolver>()};
        solver.removeObject(physic_object_id);
        solver.removeConstraint(constraint_id);
    }

    [[nodiscard]]
    float getRatio(SimulationConfiguration const& conf) const
    {
        return reserve / conf.plant_max_reserve;
    }

    void update(SimulationConfiguration const& conf, GroundCell& ground_cell, float dt)
    {
        if (reserve <= 0.0f) {
            requestRemove();
        }

        position_top += (position - position_top) * 14.0f * dt;

        float const fertilize_boost = (ground_cell.fertilize_boost * dt);
        ground_cell.fertilize_boost -= fertilize_boost;

        reserve += conf.plant_growth_rate * dt + fertilize_boost;
        if (reserve > conf.plant_max_reserve) {
            reserve = conf.plant_max_reserve;
            split_time += dt;
        } else {
            split_time = 0.0f;
        }
    }

    [[nodiscard]]
    bool splitReady(float split_cooldown) const
    {
        return split_time >= split_cooldown;
    }

private:
    void createPhysicsObject()
    {
        // Create associated physics object
        auto& solver{pez::core::getProcessor<PhysicSolver>()};
        physic_object_id = solver.createObject(position, id.instance_id);
        auto& object{solver.getObject(physic_object_id)};
        object.agent_id = id.instance_id;
        object.team     = Team::Plant;
        object.mass     = 0.2f;

        float const strength = 0.05f;
        constraint_id = static_cast<pez::core::ID>(solver.createConstraint(physic_object_id, position, strength));
    }

public:
    // State
    pez::core::ID physic_object_id = pez::core::EntityID::INVALID_ID;
    pez::core::ID constraint_id    = pez::core::EntityID::INVALID_ID;
    Vec2  position                 = {0.0f, 0.0f};
    float reserve                  = 0.0f;
    float split_time               = 0.0f;
    Vec2  direction;
    Vec2  direction_top;
    Vec2  position_top;
};


/** Plant processor
 *
 * Updates plants entities and spawn new ones
 *
 */
struct PlantUpdater : public pez::core::IProcessor
{
    SimulationConfiguration const& conf;

    Grid<GroundCell> plant;
    float            new_plant_time = 0.0f;

    PlantUpdater()
        : conf{pez::core::getSingleton<SimulationConfiguration>()}
        , plant{conf.world_size.x, conf.world_size.y}
    {}

    void update(float dt) override
    {
        auto& solver{pez::core::getProcessor<PhysicSolver>()};
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();


        std::vector<Vec2> new_plants;

        pez::core::foreach<Plant>([&](Plant& p) {
            // Fetch position
            Vec2 const target_position = solver.getObject(p.physic_object_id).position;
            p.position += (target_position - p.position) * 30.0f * dt;
            // Update growth + fertilize_boost
            p.update(conf, plant.get(target_position), dt);
            // If the plant is fully grown
            if (p.getRatio(conf) == 1.0f) {
                if (p.splitReady(conf.plant_split_cooldown)) {
                    // Get the current cell
                    IVec2 const cell_coord{
                        static_cast<int32_t>(target_position.x),
                        static_cast<int32_t>(target_position.y)
                    };

                    // Determines if a provided cell is valid
                    auto const isValid = [&](IVec2 cell){
                        bool const coord_valid = (cell.x > 0) && (cell.y > 0) && (cell.x < plant.width) && (cell.y < plant.height);
                        if (coord_valid) {
                            return !plant.get(cell).occupied;
                        }
                        return false;
                    };

                    // Pick a random 4-connected neighbour
                    IVec2 const new_cell = [&]() -> IVec2 {
                        auto const c = static_cast<uint32_t>(rng.getUnder(4.0f));
                        switch (c) {
                            case 0:
                                return {cell_coord.x - 1, cell_coord.y};
                            case 1:
                                return {cell_coord.x + 1, cell_coord.y};
                            case 2:
                                return {cell_coord.x, cell_coord.y - 1};
                            case 3:
                                return {cell_coord.x, cell_coord.y + 1};
                            default:
                                return cell_coord;
                        }
                    }();

                    // If the picked neighbour is valid, create a new plant
                    if (isValid(new_cell)) {
                        plant.get(new_cell).occupied = true;
                        p.split_time = 0.0f;
                        new_plants.push_back(static_cast<Vec2>(new_cell));
                    }
                }
            }
        });

        // Create the new plants
        for (auto& p : new_plants) {
            pez::core::create<Plant>(p, conf.plant_growth_rate * dt);
        }

        // Add random new plants
        new_plant_time += dt;
        if (new_plant_time >= conf.plants_random_new_cooldown) {
            new_plant_time -= conf.plants_random_new_cooldown;
            for (uint32_t i{conf.plants_random_new_count}; i--;) {
                Vec2 const new_plant = {
                    rng.getUnder(static_cast<float>(plant.width)),
                    rng.getUnder(static_cast<float>(plant.height)),
                };

                createPlant(new_plant, conf.plant_growth_rate * dt);
            }
        }
    }

    void createPlant(Vec2 position, float reserve)
    {
        GroundCell& cell{plant.get(position)};
        if (!cell.occupied) {
            cell.occupied = true;
            pez::core::create<Plant>(position, reserve);
        }
    }

    void clear()
    {
        for (auto& cell : plant.data) {
            cell.occupied = false;
            cell.fertilize_boost = 0.0f;
        }

        new_plant_time = 0.0f;
    }

    void createInitialEnv()
    {
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        for (uint32_t i{conf.plants_initial_count}; i--;) {
            float const margin{2.0f};
            Vec2 const new_plant = {
                rng.getRange(margin, static_cast<float>(plant.width) - margin),
                rng.getRange(margin, static_cast<float>(plant.height) - margin),
            };

            createPlant(new_plant, conf.plant_max_reserve - 1.0f);
        }
    }
};

