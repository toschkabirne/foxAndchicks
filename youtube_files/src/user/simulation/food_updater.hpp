#pragma once
#include "engine/engine.hpp"

#include "user/simulation/food.hpp"


struct FoodUpdater
{
    static void update(float dt)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& solver = pez::core::getProcessor<PhysicSolver>();

        pez::core::foreach<Food>([&](Food& f) {
            f.update(conf, dt);
            f.position = solver.objects[f.physic_object_id].position;
        });
    }
};
