#pragma once
#include "engine/engine.hpp"
#include "engine/common/number_generator.hpp"

#include "user/physics/physics.hpp"
#include "user/neat/mutator.hpp"

#include "user/simulation/agent.hpp"
#include "user/simulation/raycaster.hpp"
#include "user/simulation/agent_state_updater.hpp"
#include "user/simulation/food_updater.hpp"
#include "user/simulation/simulation_state.hpp"


/// Updates the simulation
struct Simulation : public pez::core::IProcessor
{
    /// The global configuration to be used
    SimulationConfiguration const& conf;
    /// The physics solver
    PhysicSolver&     solver;
    /// The currently selected agent (if any)
    pez::core::Handle selected;
    /// The raycaster used to fetch information in the world
    Raycaster raycaster;
    /// The simulation seed
    uint32_t seed;
    /// The number of frames that have been simulated
    uint32_t frame_count = 0;
    /// The number of alive predators
    uint32_t pred_count  = 0;
    /// The number of alive prey
    uint32_t prey_count  = 0;
    /// Checks if the simulation has already been executed once (used when the --no_restart option is used)
    bool simulation_ran_once = false;
    /// The current simulation is over
    bool simulation_over = false;
    /// Flag to tell if the simulation should automatically restart when over
    bool auto_restart = true;

    /// Constructor
    explicit
    Simulation(bool auto_restart_)
        : conf{pez::core::getSingleton<SimulationConfiguration>()}
        , solver{pez::core::getProcessor<PhysicSolver>()}
        , seed{conf.seed}
        , auto_restart{auto_restart_}
    {}

    /// Reset the simulation and starts a new one
    void restart();

    /// Create the initial prey and predators
    void createInitialPopulation() const
    {
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        // Margin used to avoid spawning entities close to the map's border
        float constexpr margin{10.0f};
        for (uint32_t i{conf.initial_pred_population}; i--;) {
            createAgent(Team::Predator,
                        {rng.getRange(margin, conf.world_size_f.x - margin),
                         rng.getRange(margin, conf.world_size_f.y - margin)});
        }

        for (uint32_t i{conf.initial_prey_population}; i--;) {
            createAgent(Team::Prey,
                        {rng.getRange(margin, conf.world_size_f.x - margin),
                         rng.getRange(margin, conf.world_size_f.y - margin)});
        }
    }

    /// Spawns an agent of the specified @p team at @p position
    static void createAgent(Team team, Vec2 position)
    {
        auto agent = pez::core::createGetHandle<Agent>(position, team);
        for (uint32_t i{10}; i--;) {
            nt::Mutator::mutateGenome(agent->genome);
        }
        agent->network = agent->genome.generateNetwork();
    }

    /// Process the current frame
    void update(float dt) override
    {
        if (!done()) {
            ++frame_count;

            AgentStateUpdater::fetchPositions();
            AgentStateUpdater::update(dt);
            FoodUpdater::update(dt);

            executeAI();
            splitAgents();
            removeDeadAgents();
        } else if (!simulation_ran_once || auto_restart) {
            restart();
            simulation_ran_once = true;
        } else {
            simulation_over = true;
        }
    }

    /// Checks if the simulation is done
    bool done()
    {
        prey_count = 0;
        pred_count = 0;

        uint32_t team_flag = 0;
        pez::core::foreach<Agent>([&](Agent const& a) {
            prey_count += (a.team == Team::Prey);
            pred_count += (a.team == Team::Predator);

            team_flag |= (1 << static_cast<uint32_t>(a.team));
        });

        uint32_t constexpr target_flag = (
            (1 << static_cast<uint32_t>(Team::Predator)) +
            (1 << static_cast<uint32_t>(Team::Prey))
        );
        return team_flag != target_flag;
    }

    /// Computes neural networks inputs and execute them
    void executeAI() const
    {
        auto& agents = pez::core::getData<Agent>().getData();
        auto& tp     = pez::core::getSingleton<tp::ThreadPool>();
        // Cannot use pez::core::parallelForeach since a vector<Ray> is needed per thread
        tp.dispatch(to<uint32_t>(agents.size()), [this, &agents](uint32_t start, uint32_t end) {
            std::vector<Raycaster::Ray> rays(conf.ray_count);
            std::vector<float>          input(conf.input_count);
            for (uint32_t i{start}; i < end; ++i) {
                Agent& a{agents[i]};
                castRays(a, rays);
                // Update eye position, could be put behind a switch to speed things up without rendering
                a.render_data.updateEyesTarget(rays);
                computeInput(a, rays, input);
                a.processInput(input);
            }
        });
    }

    /** Uses the raycaster to cast rays in the world.
     * @note This function exists because it eases the use of this functionality from outside the simulation class
     */
    void castRays(Agent const& a, std::vector<Raycaster::Ray>& rays) const
    {
        raycaster.cast(a.position, a.angle, getAgentFOV(a.team), rays, a.id.instance_id);
    }

    /// Returns the agent fov given its @p team
    [[nodiscard]]
    float getAgentFOV(Team team) const
    {
        return (team == Team::Predator) ? conf.predator_fov : conf.prey_fov;
    }

    /// Converts raw raycasting data into values suitable for the neural network
    static void computeInput(Agent const& agent, std::vector<Raycaster::Ray> const& rays, std::vector<float>& input)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        // Introspection inputs
        input[0] = agent.getHealthRatio();
        input[1] = agent.getEnergyRatio();
        input[2] = agent.getSplitRatio();
        input[3] = agent.getReserveRatio();

        // Ray related inputs
        float const ray_index_scale = 1.0f / static_cast<float>(conf.ray_count - 1);
        uint32_t current_ray = 0;
        for (uint32_t i{0}; i < conf.agent_zone_count; ++i) {
            uint32_t const current_ray_input_idx = conf.introspection_input_count + conf.zone_input_count * i;
            uint32_t const min_ray = getZoneMainRay(i, rays);
            auto const& ray = rays[min_ray];
            // Distance
            input[current_ray_input_idx + 0] = 1.0f - (ray.length / conf.agent_ray_max_dist);
            // Angle
            float const ray_ratio = to<float>(min_ray) * ray_index_scale;
            float const remapped  = 2.0f * (ray_ratio - 0.5f);
            input[current_ray_input_idx + 1] = remapped;
            // Danger score
            input[current_ray_input_idx + 2] = getTeamInput(agent.team, ray.team);
        }
    }

    /// Returns the ray of the zone @p zone_idx that will be used as input
    static uint32_t getZoneMainRay(uint32_t zone_idx, std::vector<Raycaster::Ray> const& rays)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        uint32_t const ray_idx = zone_idx * conf.agent_zone_ray_count;
        uint32_t min_idx       = ray_idx;
        float    min_length    = rays[ray_idx].length;
        for (uint32_t i{1}; i < conf.agent_zone_ray_count; ++i) {
            Raycaster::Ray const& ray = rays[ray_idx + i];
            if (ray.length < min_length) {
                min_idx    = ray_idx + i;
                min_length = ray.length;
            }
        }
        return min_idx;
    }

    /// Returns the "danger value" associated with the observed team @p ray_team for entity of team @p agent_team
    static float getTeamInput(Team agent_team, Team ray_team)
    {
        if (agent_team == Team::Prey) {
            switch (ray_team) {
                case Team::Predator:
                    return -1.0f;
                case Team::Prey:
                    return 0.25f;
                case Team::Plant:
                    return 1.0f;
                default:
                    return 0.0f;
            }
        } else if (agent_team == Team::Predator) {
            switch (ray_team) {
                case Team::Food:
                    return 1.0f;
                case Team::Prey:
                    return 0.5f;
                case Team::Predator:
                    return 0.25f;
                default:
                    return 0.0f;
            }
        }
        return 0.0f;
    }

    /// Performs agent splitting for reproduction
    static void splitAgents()
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        std::vector<pez::core::ID> to_split;

        pez::core::foreach<Agent>([&to_split](Agent& agent){
            if (agent.metabolism.splitReady()) {
                agent.metabolism.performSplit();
                ++agent.split_count;
                to_split.push_back(agent.id.instance_id);
            }
        });

        for (auto id : to_split) {
            auto a = pez::core::getHandle<Agent>(id);
            Vec2 const child_offset{rng.getRange(1.0f), rng.getRange(1.0f)};
            auto child = pez::core::createGetHandle<Agent>(a->position + child_offset, a->team);
            child->inheritGenome(a->genome);
            child->metabolism.reserve = 0.5f * a->metabolism.split_requirement;
        }
    }

    /// Prune dead agents from the simulation
    static void removeDeadAgents()
    {
        pez::core::foreach<Agent>([](Agent& agent){
            if (agent.isDead()) {
                agent.requestRemove();
            }
        });
    }

    /// Select the agent at the provided @p position in the world
    void selectAgent(Vec2 position)
    {
        // This allows to retrieve the closest agent from the mouse
        float select_radius = 1.0f;
        selected = {};

        pez::core::foreach<Agent>([&](Agent const& agent) {
            Vec2 const v{position - agent.position};
            float const dist{MathVec2::length(v)};
            if (dist < select_radius) {
                selected = pez::core::createEntityHandle(agent.id);
                select_radius = dist;
            }
        });
    }
};