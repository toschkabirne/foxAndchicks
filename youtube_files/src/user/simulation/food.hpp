#pragma once
#include "engine/engine.hpp"

#include "user/physics/physics.hpp"
#include "user/neat/genome.hpp"
#include "user/neat/mutator.hpp"
#include "user/physics/spring_particle.hpp"
#include "user/simulation/agent_render_data.hpp"
#include "user/simulation/plant.hpp"

#include "team.hpp"


struct Food : public pez::core::Entity
{
public:
    Food() = default;

    explicit
    Food(pez::core::EntityID id_, Vec2 position_, float reserve_)
        : pez::core::Entity{id_}
        , position{position_}
        , team{Team::Food}
        , reserve{reserve_}
        , reserve_init{reserve_}
    {
        createPhysicsObject();
    }

    void onRemove() override
    {
        pez::core::getProcessor<PhysicSolver>().removeObject(physic_object_id);
    }

    [[nodiscard]]
    float getRatio(SimulationConfiguration const& conf) const
    {
        return reserve / conf.agent_reserve_max;
    }

    void update(SimulationConfiguration const& conf, float dt)
    {
        reserve -= conf.food_decay * dt;
        if (reserve <= 0.0f) {
            requestRemove();
            auto& plant_updater = pez::core::getProcessor<PlantUpdater>();
            plant_updater.plant.get(position).fertilize_boost += reserve_init;
        }
    }

private:
    void createPhysicsObject()
    {
        // Create associated physics object
        auto& solver{pez::core::getProcessor<PhysicSolver>()};
        physic_object_id = solver.createObject(position, id.instance_id);
        auto& object{solver.getObject(physic_object_id)};
        object.agent_id = id.instance_id;
        object.team     = team;
    }

public:
    Team team = Team::Food;

    // State
    pez::core::ID physic_object_id = pez::core::EntityID::INVALID_ID;
    Vec2  position                 = {0.0f, 0.0f};
    float reserve                  = 0.0f;
    float reserve_init             = 0.0f;
};
