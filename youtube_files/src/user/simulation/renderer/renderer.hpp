#pragma once
#include "engine/engine.hpp"
#include "engine/common/thread_pool/thread_pool.hpp"
#include "engine/common/card/card.hpp"
#include "engine/common/cooldown.hpp"

#include "user/physics/physics.hpp"
#include "user/neat/network_renderer.hpp"
#include "user/simulation/agent.hpp"

#include "graph_widget.hpp"
#include "background_grid.hpp"
#include "debug_grid.hpp"
#include "gauge.hpp"
#include "./agent_viewer.hpp"
#include "./minimap.hpp"
#include "./time_state.hpp"
#include "./observer.hpp"
#include "./exit_prompt.hpp"

#include "user/logo/logo.c"


/// Renders the simulation
struct Renderer : public pez::core::IRenderer
{
    static constexpr float    graph_width        = 500.0f;
    static constexpr float    graph_height       = 200.0f;
    static constexpr uint32_t graph_data_count   = 1000;
    static constexpr uint32_t graph_x_period     = 200;
    static constexpr float    graph_update_delay = 0.1f;
    static constexpr float    background_margin  = 8.0f;

    SimulationConfiguration const& conf;

    Card            background;
    BackgroundGrid  grid;

    // Agents rendering
    sf::VertexArray agent_va;
    sf::VertexArray light_va;
    sf::VertexArray eyes_va;

    // Food
    sf::VertexArray food_va;
    sf::VertexArray plant_va;

    sf::Texture* circle_solid_texture;
    sf::Texture* agent_solid_texture;
    sf::Texture* circle_empty_texture;
    sf::Texture* shadow_texture;
    sf::Texture* plant_texture;
    sf::Texture* light_texture;

    tp::ThreadPool&  thread_pool;
    PhysicSolver&    solver;

    Cooldown    population_update_cooldown;
    GraphWidget population_prey;
    GraphWidget population_pred;
    GraphWidget count_food;
    GraphWidget count_plant;

    AgentViewer agent_viewer;
    Minimap     minimap;
    TimeState   time_state;

    sf::Text grid_text;

    sf::Image   logo_image;
    sf::Texture logo_texture;
    sf::Text    logo_text;

    bool enable_render = true;
    bool render_graphs = true;

    Observer    observer;
    bool        auto_obs = false;
    SmoothVec2  auto_focus;
    SmoothFloat auto_zoom;

    std::optional<Observer::Event> last_event;
    bool zoom_done  = true;
    bool zoom_first = false;

    ExitPrompt exit_prompt;
    
    Renderer();

    void render(pez::render::Context& context) override;

    void renderAgents(pez::render::Context& context);

    void addAgentToVertexArray(const Agent& a, uint32_t index);

    void addAgentEyes(const AgentRenderData& render_info, Vec2 dir, Vec2 position, uint32_t index, Vec2 elongation);

    void addOneEye(Vec2 position, Vec2 look_at, float dist, uint32_t index);

    void renderFood(pez::render::Context& context);

    void renderPlant(pez::render::Context& context);

    void toggleRender();

    void loadObserver(std::string const& filename, bool activate)
    {
        observer.loadFromFile(filename);
        auto_obs = activate;
    }

    void addEvent()
    {
        auto const& simulation = pez::core::getProcessor<Simulation>();
        if (pez::core::isValidHandle(simulation.selected)) {
            observer.addEvent(simulation.frame_count, simulation.selected.id.instance_id, pez::render::getZoom());
        } else {
            observer.addEvent(simulation.frame_count, pez::render::getFocus(), pez::render::getZoom());
        }
    }
};
