#pragma once
#include "user/simulation/simulation.hpp"
#include "user/simulation/renderer/agent_stats_viewer.hpp"


struct AgentViewer
{
    SimulationConfiguration const& conf;

    NetworkRenderer   network_renderer;
    AgentStatsViewer  agent_stats_viewer;
    pez::core::Handle selected_last;

    // Draw the perception rays of the selected agent
    sf::VertexArray debug_rays_va;
    sf::VertexArray debug_main_rays_va;
    std::vector<Raycaster::Ray> rays;

    AgentViewer()
        : conf{pez::core::getSingleton<SimulationConfiguration>()}
        , debug_rays_va{sf::PrimitiveType::Lines, 2 * conf.ray_count}
        , rays{conf.ray_count}
        , debug_main_rays_va{sf::PrimitiveType::Triangles, 3 * conf.agent_zone_count}
        , agent_stats_viewer{conf}
    {
        Vec2 const ui_scale = {conf.ui_scale, conf.ui_scale};

        network_renderer.setScale(ui_scale);
        network_renderer.setPosition({consts::ui_margin, consts::ui_margin});

        network_renderer.labels.emplace_back("Health");
        network_renderer.labels.emplace_back("Energy");
        network_renderer.labels.emplace_back("Split");
        network_renderer.labels.emplace_back("Reserve");

        for (uint32_t i{0}; i < conf.agent_zone_count; ++i) {
            network_renderer.labels.push_back("Zone " + toString(i) + " dist");
            network_renderer.labels.push_back("Zone " + toString(i) + " angle");
            network_renderer.labels.push_back("Zone " + toString(i) + " attract");
        }

        agent_stats_viewer.setScale(ui_scale);
    }

    void renderRays(pez::render::Context& context)
    {
        auto const& simulation = pez::core::getProcessor<Simulation>();

        // Ensure the selected agent exists
        if (pez::core::isValid<Agent>(simulation.selected)) {
            Agent const& agent{pez::core::get<Agent>(simulation.selected)};
            simulation.castRays(agent, rays);
            for (uint32_t i{0}; i < conf.ray_count; ++i) {
                auto const& ray{rays[i]};
                debug_rays_va[2 * i    ].position = agent.position;
                debug_rays_va[2 * i + 1].position = agent.position + ray.direction * ray.length;

                sf::Color color = sf::Color::White;
                if (ray.team == Team::Predator) {
                    color = conf.pred_color;
                } else if (ray.team == Team::Prey) {
                    color = conf.prey_color;
                } else if (ray.team == Team::Food) {
                    color = conf.food_color;
                } else if (ray.team == Team::Plant) {
                    color = conf.plant_color;
                }
                debug_rays_va[2 * i    ].color = color;
                debug_rays_va[2 * i + 1].color = color;
            }
            context.draw(debug_rays_va);

            for (uint32_t i{0}; i < conf.agent_zone_count; ++i) {
                renderZone(i, agent, context);
            }
            context.draw(debug_main_rays_va);
        }
    }

    void renderSelected(pez::render::Context& context)
    {
        auto const& simulation = pez::core::getProcessor<Simulation>();

        if (pez::core::isValid<Agent>(simulation.selected)) {
            Agent const& agent{pez::core::get<Agent>(simulation.selected)};

            // If it is a newly selected one, initialize widgets
            if (simulation.selected != selected_last) {
                selected_last = simulation.selected;
                network_renderer.initialize(agent.network);

                float const current_y = network_renderer.background.getOutlineSize().y * conf.ui_scale + network_renderer.position.y + consts::ui_margin;
                agent_stats_viewer.setPosition(consts::ui_margin, current_y);
            }

            network_renderer.update(agent.network);
            context.drawDirect(network_renderer);

            agent_stats_viewer.updateStats(agent);
            context.drawDirect(agent_stats_viewer);
        }
    }

    void renderZone(uint32_t zone_idx, Agent const& agent, pez::render::Context& context)
    {
        constexpr float arc_thickness = 0.15f;

        uint32_t const        zone_ray_start    = zone_idx * conf.agent_zone_ray_count;
        uint32_t const        zone_main_ray_idx = Simulation::getZoneMainRay(zone_idx, rays);
        Raycaster::Ray const& zone_main_ray     = rays[zone_main_ray_idx];
        sf::Color const       color             = getTeamColor(zone_main_ray.team, conf);

        sf::VertexArray va_outline{sf::PrimitiveType::TriangleStrip, conf.agent_zone_ray_count * 2};
        sf::VertexArray va        {sf::PrimitiveType::TriangleFan,   conf.agent_zone_ray_count + 1};

        // Generate main ray geometry
        uint32_t const va_idx = zone_idx * 3;
        debug_main_rays_va[va_idx + 0].position = agent.position;
        debug_main_rays_va[va_idx + 0].color    = {color.r, color.g, color.b};

        if (zone_main_ray.team != Team::None) {
            Vec2 const tip = agent.position + zone_main_ray.direction * zone_main_ray.length;
            Vec2 const nrm = MathVec2::normal(zone_main_ray.direction) * 0.25f * arc_thickness;
            debug_main_rays_va[va_idx + 1].position = tip + nrm;
            debug_main_rays_va[va_idx + 1].color = color;
            debug_main_rays_va[va_idx + 2].position = tip - nrm;
            debug_main_rays_va[va_idx + 2].color = color;
        } else {
            debug_main_rays_va[va_idx + 1].position = agent.position;
            debug_main_rays_va[va_idx + 2].position = agent.position;
        }

        for (uint32_t i{0}; i < conf.agent_zone_ray_count; ++i) {
            uint32_t const ray_idx = zone_ray_start + i;
            va_outline[2 * i + 0].position = agent.position + rays[ray_idx].direction * zone_main_ray.length;
            va_outline[2 * i + 1].position = agent.position + rays[ray_idx].direction * (zone_main_ray.length - arc_thickness);
            va_outline[2 * i + 0].color = color;
            va_outline[2 * i + 1].color = color;

            va[i].position = agent.position + rays[ray_idx].direction * zone_main_ray.length;
            va[i].color    = {color.r, color.g, color.b, 100};
        }
        va[conf.agent_zone_ray_count].position = agent.position;
        va[conf.agent_zone_ray_count].color    = {color.r, color.g, color.b, 0};

        context.draw(va);
        context.draw(va_outline);
    }

    static sf::Color getTeamColor(Team team, SimulationConfiguration const& conf)
    {
        switch (team) {
            case Team::Predator:
                return conf.pred_color;
            case Team::Prey:
                return conf.prey_color;
            case Team::Food:
                return conf.food_color;
            case Team::Plant:
                return conf.plant_color;
            default:
                return sf::Color::White;
        }
    }
};