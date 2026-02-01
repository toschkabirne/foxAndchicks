#pragma once
#include "engine/engine.hpp"
#include <map>
#include <sstream>
#include <string>
#include <fstream>

#include "ConfigurationLoader/include/configuration_loader.hpp"

// This macro works only for parameters having same name and type with the txt one
#define LOAD_PARAMETER(P) loadInto(loader, #P, &P)

struct SimulationConfiguration
{
    // Simulation
    uint32_t seed = 9;
    IVec2    world_size{300, 300};
    Vec2     world_size_f{static_cast<Vec2>(world_size)};
    uint32_t initial_prey_population = 1000;
    uint32_t initial_pred_population = 1000;

    // Render
    sf::Color const prey_color{17, 138, 178};
    sf::Color const pred_color{239, 71, 111};
    sf::Color const food_color{255, 209, 102};
    sf::Color const plant_color{6, 214, 160};

    // Neural Network
    uint32_t agent_zone_count      = 5;
    uint32_t agent_zone_ray_count  = 12;

    uint32_t const zone_input_count          = 3;
    uint32_t const introspection_input_count = 4; // Health, Energy, Split, Reserve

    uint32_t input_count  = zone_input_count * agent_zone_count + introspection_input_count;
    uint32_t output_count = 2;
    uint32_t ray_count    = agent_zone_ray_count * agent_zone_count;

    // Agents
    float agent_ray_max_dist       = 100.0f;
    float agent_speed_max          = 2.0f;
    float agent_angular_speed_max  = 270.0f;
    float agent_reserve_decay      = 1.0f;
    float agent_speed_penalty_time = 2.0f;

    float predator_attack_threshold  = 0.5f;
    float predator_attack_cooldown   = 0.25f;
    float predator_attack_damage     = 75.0f;
    float predator_eat_rate          = 80.0f;
    float predator_fov               = 90.0f;
    float predator_health_decay      = 1.0f;
    float predator_split_requirement = 35.0f;

    float prey_attack_threshold     = 0.7f;
    float prey_attack_cooldown      = 1.0f;
    float prey_attack_damage        = 20.0f;
    float prey_eat_rate             = 60.0f;
    float prey_fov                  = 315.0f;
    float prey_health_decay         = 2.0f;
    float prey_split_requirement    = 35.0f;

    // Metabolism
    float agent_health_max    = 100.0f;
    float agent_reserve_max   = 150.0f;
    float agent_metabolism_flow = 10.0f;
    float agent_energy_max    = 100.0f;

    // Mutations
    float offset_bias_proba   = 0.2f;
    float offset_weight_proba = 0.5f;
    float new_node_proba      = 0.05f;
    float new_conn_proba      = 0.7f;
    float new_value_proba     = 0.1f;
    float weight_range        = 2.0f;
    float weight_small_range  = 0.1f;

    // Food
    float food_decay = 1.5f;

    // Plants
    uint32_t plants_initial_count       = 1000;
    float    plants_random_new_cooldown = 5.0f;
    uint32_t plants_random_new_count    = 50;

    float    plant_max_reserve          = 50.0f;
    float    plant_growth_rate          = 10.0f;
    float    plant_split_cooldown       = 3.0f;

    // UI
    float ui_scale = 1.0f;

    explicit
    SimulationConfiguration(cload::ConfigurationLoader const& loader)
    {
        // Ensure the provided loader is valid
        if (!loader.isValid()) {
            std::cout << "Could not open configuration file, using default values." << std::endl;
        } else {

        // Compute UI scale
        Vec2 window_size{2560, 1440};
        loader.tryReadSequenceIntoArray<2>("window_size", &window_size.x);
        // The UI has been designed for a 1440p screen
        ui_scale = window_size.y / 1440.0f;

        LOAD_PARAMETER(seed);
        LOAD_PARAMETER(initial_prey_population);
        LOAD_PARAMETER(initial_pred_population);
        int32_t world_side_size;
        loadInto(loader, "world_size", &world_side_size);
        world_size = {world_side_size, world_side_size};
        world_size_f = static_cast<Vec2>(world_size);

        // Plants
        LOAD_PARAMETER(plants_initial_count);
        LOAD_PARAMETER(plants_random_new_cooldown);
        LOAD_PARAMETER(plants_random_new_count);
        LOAD_PARAMETER(plant_split_cooldown);

        // Agents
        LOAD_PARAMETER(predator_attack_damage);
        LOAD_PARAMETER(prey_attack_damage);

#ifndef DEMO
        LOAD_PARAMETER(agent_health_max);
        LOAD_PARAMETER(agent_reserve_decay);
        LOAD_PARAMETER(agent_reserve_max);
        LOAD_PARAMETER(agent_metabolism_flow);
        LOAD_PARAMETER(agent_energy_max);
        LOAD_PARAMETER(agent_zone_count);
        LOAD_PARAMETER(agent_zone_ray_count);
        LOAD_PARAMETER(agent_ray_max_dist);
        LOAD_PARAMETER(agent_speed_max);
        LOAD_PARAMETER(agent_angular_speed_max);
        LOAD_PARAMETER(agent_speed_penalty_time);

        LOAD_PARAMETER(predator_attack_threshold);
        LOAD_PARAMETER(predator_attack_cooldown);
        LOAD_PARAMETER(predator_eat_rate);
        LOAD_PARAMETER(predator_fov);
        LOAD_PARAMETER(predator_split_requirement);
        LOAD_PARAMETER(predator_health_decay);

        LOAD_PARAMETER(prey_attack_threshold);
        LOAD_PARAMETER(prey_attack_cooldown);
        LOAD_PARAMETER(prey_eat_rate);
        LOAD_PARAMETER(prey_fov);
        LOAD_PARAMETER(prey_split_requirement);
        LOAD_PARAMETER(prey_health_decay);

        LOAD_PARAMETER(food_decay);
        LOAD_PARAMETER(plant_max_reserve);
        LOAD_PARAMETER(plant_growth_rate);

        LOAD_PARAMETER(offset_bias_proba);
        LOAD_PARAMETER(offset_weight_proba);
        LOAD_PARAMETER(new_node_proba);
        LOAD_PARAMETER(new_conn_proba);
        LOAD_PARAMETER(new_value_proba);
        LOAD_PARAMETER(weight_range);
        LOAD_PARAMETER(weight_small_range);

        // Update ray count and network size
        input_count  = zone_input_count * agent_zone_count + introspection_input_count;
        output_count = 2;
        ray_count    = agent_zone_ray_count * agent_zone_count;
#endif
        }
        
        // Convert degree to radians
        agent_angular_speed_max = Math::degToRad(agent_angular_speed_max);
        predator_fov      = Math::degToRad(predator_fov);
        prey_fov          = Math::degToRad(prey_fov);
    }

    template<typename TType>
    void loadInto(cload::ConfigurationLoader const& loader, std::string const& key, TType* attribute)
    {
        if (loader.tryReadValueInto(key, attribute)) {
            std::cout << '"' << key << "\" loaded with value: " << *attribute << std::endl;
        } else {
            std::cout << "[WARNING] \"" << key << "\" could not be loaded, using default value of " << *attribute << std::endl;
        }
    }

    template<>
    void loadInto<IVec2>(cload::ConfigurationLoader const& loader, std::string const& key, IVec2* attribute)
    {
        if (loader.tryReadSequenceIntoArray<2>(key, &attribute->x)) {
            std::cout << '"' << key << "\" loaded with value: [" << attribute->x << ", " << attribute->y << "]" << std::endl;
        } else {
            std::cout << "[WARNING] \"" << key << "\" could not be loaded, using default value of [" << attribute->x << ", " << attribute->y << "]" << std::endl;
        }
    }

    template<>
    void loadInto<Vec2>(cload::ConfigurationLoader const& loader, std::string const& key, Vec2* attribute)
    {
        if (loader.tryReadSequenceIntoArray<2>(key, &attribute->x)) {
            std::cout << '"' << key << "\" loaded with value: [" << attribute->x << ", " << attribute->y << "]" << std::endl;
        } else {
            std::cout << "WARNING, \"" << key << "\" could not be loaded, using default value of [" << attribute->x << ", " << attribute->y << "]" << std::endl;
        }
    }
};
