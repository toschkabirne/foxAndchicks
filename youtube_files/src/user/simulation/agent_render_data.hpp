#pragma once
#include "engine/engine.hpp"
#include "engine/common/smooth/smooth_value.hpp"

#include "user/physics/spring_particle.hpp"
#include "user/configuration.hpp"
#include "user/simulation/raycaster.hpp"


struct AgentRenderData
{
    float          radius              = 0.5f;
    sf::Color      color;
    SpringParticle spring_length;
    SpringParticle spring_width;
    Vec2           look_at             = {};
    Vec2           target_look_at      = {};
    float          look_at_dist        = 1.0f;
    float          look_at_dist_target = 1.0f;
    SmoothFloat    angle               = {0.0f, 30.0f};
    Vec2           dir                 = {};

    void setElongation(PhysicObject const& obj)
    {
        const float object_speed = obj.getSpeed();
        // Elongate length
        spring_length.addX(1.5f * object_speed);
        spring_width.addX(-1.25f * object_speed);
    }

    void update(float current_angle, float dt)
    {
        angle = current_angle;
        dir = {std::cos(angle), std::sin(angle)};
        // Elongate length with speed
        spring_length.update(dt);
        // Contract width
        spring_width.update(dt);
        // Update look
        const float look_speed = 8.0f;
        look_at      += (target_look_at - look_at) * (look_speed * dt);
        look_at_dist += (look_at_dist_target - look_at_dist) * (1.5f * look_speed * dt);
    }

    void updateEyesTarget(std::vector<Raycaster::Ray> const& rays)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        float const dist_norm = 1.0f / (conf.agent_ray_max_dist * 0.5f);
        float       min_dist  = 2.0f * dist_norm * conf.agent_ray_max_dist;

        size_t const ray_count = rays.size();
        for (uint32_t i{0}; i < ray_count; ++i) {
            float const d = rays[i].length * dist_norm;
            if (d < min_dist) {
                target_look_at      = rays[i].direction;
                look_at_dist_target = d;
                min_dist            = d;
            }
        }
        look_at_dist_target = std::min(1.0f, look_at_dist_target);
    }
};
