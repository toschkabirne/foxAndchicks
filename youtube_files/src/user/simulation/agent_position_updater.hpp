#pragma once
#include "engine/engine.hpp"

#include "agent.hpp"
#include "user/physics/physics.hpp"


struct AgentPositionUpdater : public pez::core::IProcessor
{
    void update(float dt) override
    {
        auto& solver = pez::core::getProcessor<PhysicSolver>();
        pez::core::parallelForeach<Agent>([&solver, dt](Agent& agent) {
            // Compute new angle and direction
            agent.updateDirection(dt);
            // Update physics object's accordingly
            /// @TODO maybe take this into account for physics?
            // Compute the mass of the agent, the more it stores, the more it weights
            float const agent_mass = 1.0f + agent.getReserveRatio() + agent.getSplitRatio();
            // Compute how much energy it would require for the agent to move
            float const requested_energy = (0.5f * agent_mass * agent.speed * agent.speed) * dt;
            if (requested_energy > 0.0f) {
                // Compute how much of the requested energy the agent can provide
                float const available = std::min(agent.metabolism.energy, requested_energy);
                float const ratio = available / requested_energy;
                // Consume the energy
                agent.metabolism.energy -= available;
                // Actually move the agent
                PhysicObject& physic_object = solver.objects[agent.physic_object_id];
                physic_object.position += agent.direction * (agent.speed * dt * ratio);
            }
        });
    }
};