#pragma once
#include "engine/engine.hpp"
#include "agent.hpp"

#include "user/physics/physics.hpp"
#include "user/simulation/plant.hpp"


struct AgentStateUpdater
{
    /// Predator specific update
    static void updatePredator(Agent& predator, float dt)
    {
        // Perform base update (metabolism, cooldown)
        auto const& solver{pez::core::getProcessor<PhysicSolver>()};
        // Loop over all colliding objects
        solver.foreachCollider(predator.position, [&](PhysicObject const& object, Vec2 n) {
            // If there is food, check if it can be eaten
            if (object.team == Team::Food) {
                return AgentStateUpdater::checkFood(predator, pez::core::get<Food>(object.agent_id), n, dt);
            }
            // If there is a Prey, process the encounter
            else if (object.team == Team::Prey) {
                return AgentStateUpdater::processEncounter(predator, pez::core::get<Agent>(object.agent_id), n);
            }
            // Continue iterating over colliders if nothing
            return false;
        });
    }

    /// Prey specific update
    static void updatePrey(Agent& prey, float dt)
    {
        // Loop over all colliding objects
        auto const& solver{pez::core::getProcessor<PhysicSolver>()};
        solver.foreachCollider(prey.position, [&](PhysicObject const& object, Vec2 n) {
            // If there is food, check if it can be eaten
            if (object.team == Team::Plant) {
                return AgentStateUpdater::checkPlantEat(prey, pez::core::get<Plant>(object.agent_id), n, dt);
            }
            // Continue iterating over colliders if nothing
            return false;
        });
    }

    /** Synchronizes the agent's position with its associated physics object
     *
     * @TODO This may be merged into the update step
     */
    static void fetchPositions()
    {
        auto const& solver = pez::core::getProcessor<PhysicSolver>();
        pez::core::parallelForeach<Agent>([&solver](Agent& agent) {
            PhysicObject const& obj{solver.getObject(agent.physic_object_id)};
            agent.position = obj.position;
            agent.render_data.setElongation(obj);
        });
    }

    /** Updates all agents
     *
     * @param dt The time step
     */
    static void update(float dt)
    {
        // These operations can be done in parallel since they are independent
        pez::core::parallelForeach<Agent>([dt](Agent& agent) {
            agent.baseUpdate(dt);
            agent.updateRenderData(dt);
        });

        // These ones cannot because they would be data races
        pez::core::foreach<Agent>([dt](Agent& agent) {
            if (agent.team == Team::Predator) {
                AgentStateUpdater::updatePredator(agent, dt);
            } else if (agent.team == Team::Prey) {
                AgentStateUpdater::updatePrey(agent, dt);
            }
        });
    }

    /** Process an encounter between a Predator and a Prey
     *
     * @param predator The Predator
     * @param prey The Prey
     * @param n The normalized direction going from Predator to Prey
     *
     * @return True if the Predator hit something
     */
    static bool processEncounter(Agent& predator, Agent& prey, Vec2 n)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        bool hit = false;
        // Check that predator can attack prey
        float const dot_predator = MathVec2::dot(predator.direction, n);
        if (dot_predator >= conf.predator_attack_threshold) {
            // Prey is in attach FOV
            if (predator.canAttack()) {
                // Predator can attack (attack cooldown is ready)
                predator.attack();
                predator.attack_cooldown = conf.predator_attack_cooldown;
                prey.addDamage(conf.predator_attack_damage);
                hit = true;
                if (prey.isDead()) {
                    // The Prey is killed
                    /// @TODO parametrize 0.25f
                    predator.metabolism.addToReserve(0.25f * prey.metabolism.reserve);
                    ++predator.kill_count;
                }
            }
        }

        // Check that prey can attack predator
        float const dot_prey = MathVec2::dot(prey.direction, -n);
        if (dot_prey >= conf.prey_attack_threshold) {
            // Predator is in attach FOV
            if (prey.canAttack()) {
                // Prey can attack (attack cooldown is ready)
                predator.attack();
                prey.attack_cooldown = conf.prey_attack_cooldown;
                predator.addDamage(conf.prey_attack_damage);
                if (predator.isDead()) {
                    ++prey.kill_count;
                }
            }
        }

        return hit;
    }

    /** Checks that the Predator can eat the Food it is colliding
     *
     * @param predator The Predator
     * @param food The Food
     * @param n The normalized direction going from Predator to Food
     * @param dt The time step
     */
    static bool checkFood(Agent& predator, Food& food, Vec2 n, float dt)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        float constexpr food_coef = 1.0f;
        // Check that predator can attack prey
        float const dot_predator = MathVec2::dot(predator.direction, n);
        if (dot_predator >= conf.predator_attack_threshold) {
            float const eat_amount = std::min(conf.predator_eat_rate * dt, std::max(0.0f, food.reserve));
            food.reserve -= food_coef * eat_amount;
            predator.metabolism.addToReserve(eat_amount);
            return true;
        }
        return false;
    }

    /**
     *
     * @param prey
     * @param plant
     * @param n
     * @param dt
     * @return True if the Prey hit something
     */
    static bool checkPlantEat(Agent& prey, Plant& plant, Vec2 n, float dt)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        float constexpr food_coef = 1.0f;
        // Check that predator can attack prey
        float const dot_prey = MathVec2::dot(prey.direction, n);
        if (dot_prey >= conf.prey_attack_threshold) {
            float const eat_amount = std::min(conf.prey_eat_rate * dt, std::max(0.0f, plant.reserve));
            plant.reserve -= food_coef * eat_amount;
            prey.metabolism.addToReserve(eat_amount);
            return true;
        }
        return false;
    }
};
