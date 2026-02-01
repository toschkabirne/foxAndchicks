#pragma once
#include "engine/engine.hpp"

#include "user/physics/physics.hpp"
#include "user/neat/genome.hpp"
#include "user/neat/mutator.hpp"
#include "user/physics/spring_particle.hpp"
#include "user/simulation/agent_render_data.hpp"
#include "user/simulation/simulation_configuration.hpp"

#include "food.hpp"
#include "metabolism.hpp"
#include "team.hpp"


struct Agent : public pez::core::Entity
{
public:
    explicit
    Agent(pez::core::EntityID id_, Vec2 position_, Team team_)
        : pez::core::Entity{id_}
        , conf{&pez::core::getSingleton<SimulationConfiguration>()}
        , position{position_}
        , team{team_}
        , genome{conf->input_count, conf->output_count}
        , metabolism{getHealthDecay(team_, *conf), getSplitRequirement(team_, *conf)}
    {
        //genome.getOutput(1).activation = nt::Activation::Sigm;
        createPhysicsObject();
        render_data.color = (team == Team::Predator) ? conf->pred_color : conf->prey_color;
    }

    void onRemove() override
    {
        pez::core::getProcessor<PhysicSolver>().removeObject(physic_object_id);
        // Create food source
        pez::core::create<Food>(position, metabolism.reserve + metabolism.split);
    }

    void processInput(std::vector<float> const& input)
    {
        float constexpr speed_penalty_ratio = 0.25f;
        if (network.execute(input)) {
            auto const& output = network.output;
            angular_speed = output[0] * conf->agent_angular_speed_max;
            if (speed_penalty_time == 0.0f) {
                speed = output[1] * conf->agent_speed_max;
            } else {
                speed = output[1] * conf->agent_speed_max * speed_penalty_ratio;
            }
        }
    }

    void inheritGenome(nt::Genome const& parent_genome)
    {
        genome = parent_genome;
        nt::Mutator::mutateGenome(genome);
        createNetwork();
    }

    void updateDirection(float dt)
    {
        angle += angular_speed * dt;
        direction = {std::cos(angle), std::sin(angle)};
    }

    [[nodiscard]]
    bool canAttack() const
    {
        return attack_cooldown <= 0.0f;
    }

    void updateRenderData(float dt)
    {
        render_data.update(angle, dt);
    }

    [[nodiscard]]
    float getHealth() const
    {
        return metabolism.health;
    }

    [[nodiscard]]
    float getHealthRatio() const
    {
        return getHealth() / conf->agent_health_max;
    }

    [[nodiscard]]
    float getSplit() const
    {
        return metabolism.split;
    }

    [[nodiscard]]
    float getSplitRatio() const
    {
        return getSplit() / metabolism.split_requirement;
    }

    [[nodiscard]]
    float getEnergy() const
    {
        return metabolism.energy;
    }

    [[nodiscard]]
    float getEnergyRatio() const
    {
        return getEnergy() / conf->agent_energy_max;
    }

    [[nodiscard]]
    float getReserve() const
    {
        return metabolism.reserve;
    }

    [[nodiscard]]
    float getReserveRatio() const
    {
        return getReserve() / conf->agent_reserve_max;
    }

    void addHealth(float quantity)
    {
        metabolism.addHealth(quantity);
    }

    void addDamage(float dmg)
    {
        metabolism.addHealth(-dmg);
        speed_penalty_time = conf->agent_speed_penalty_time;
        render_data.spring_width.addX(-0.5f);
        render_data.spring_length.addX(-0.5f);
    }

    void baseUpdate(float dt)
    {
        speed_penalty_time -= dt;
        if (speed_penalty_time < 0.0f) {
            speed_penalty_time = 0.0f;
        }
        attack_cooldown -= dt;
        metabolism.update(dt);
    }

    [[nodiscard]]
    bool isDead() const
    {
        return metabolism.health <= 0.0f;
    }

    void attack()
    {
        render_data.spring_width.addX(-0.25f);
        render_data.spring_length.addX(0.75f);
    }

    [[nodiscard]]
    static float getHealthDecay(Team team, SimulationConfiguration const& conf)
    {
        if (team == Team::Predator) {
            return conf.predator_health_decay;
        }
        if (team == Team::Prey) {
            return conf.prey_health_decay;
        }
        // Should not happen
        return 0.0f;
    }

    [[nodiscard]]
    static float getSplitRequirement(Team team, SimulationConfiguration const& conf)
    {
        if (team == Team::Predator) {
            return conf.predator_split_requirement;
        }
        if (team == Team::Prey) {
            return conf.prey_split_requirement;
        }
        // Should not happen
        return 0.0f;
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

    void createNetwork()
    {
        network = genome.generateNetwork();
    }

public:
    SimulationConfiguration const* conf;

    Team team = Team::None;

    // State
    pez::core::ID physic_object_id = pez::core::EntityID::INVALID_ID;
    Vec2          position         = {0.0f, 0.0f};
    float         angle            = 0.0f;
    float         angular_speed    = 0.0f;
    Vec2          direction        = {1.0f, 0.0f};
    float         speed            = 0.0f;
    float         attack_cooldown  = 0.0f;
    uint32_t      kill_count       = 0;
    uint32_t      split_count      = 0;

    float speed_penalty_time = 0.0f;

    // Genetic
    nt::Genome  genome;
    nt::Network network;

    // Metabolism
    Metabolism metabolism;

    // Render
    AgentRenderData render_data;
};
