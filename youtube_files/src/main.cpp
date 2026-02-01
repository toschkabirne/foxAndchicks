#include <filesystem>

#include "engine/window/window_context_handler.hpp"
#include "engine/common/racc.hpp"

#include "user/configuration.hpp"
#include "user/physics/physics.hpp"

#include "user/simulation/simulation.hpp"
#include "user/simulation/simulation_state.hpp"
#include "user/simulation/simulation_configuration.hpp"
#include "user/simulation/agent_position_updater.hpp"
#include "user/simulation/renderer/renderer.hpp"
#include "user/simulation/plant.hpp"
#include "user/simulation/food.hpp"


int main(int argc, char* argv[])
{
    auto const exe_path = argv[0];
    auto const containing_folder = std::filesystem::path{exe_path}.parent_path();

    // Load window configuration
    cload::ConfigurationLoader loader(containing_folder.string() + "/conf.txt");
    UVec2 window_size{2560, 1440};
    loader.tryReadSequenceIntoArray<2>("window_size", &window_size.x);
    auto const fullscreen = loader.tryGetValueAs<bool>("fullscreen").value_or(true);

    // Load thread_count
    uint32_t thread_count = 0;
    loader.tryReadValueInto("thread_count", &thread_count);

    // Initialize Engine and its sub systems
    pez::core::createSystems(thread_count);

#ifdef DEMO
    std::string const app_title = "Predator vs Prey - Demo";
#else
    std::string const app_title = "Predator vs Prey";
#endif

    // Check app parameters
    bool headless = false;
    bool auto_restart = true;
    for (int32_t i{1}; i < argc; ++i) {
        if (std::string(argv[i]) == "--headless") {
            headless = true;
        } else if (std::string(argv[i]) == "--no_restart") {
            auto_restart = false;
        }
    }
    std::unique_ptr<pez::render::WindowContextHandler> app_ptr;
    // Create window if not in headless mode
    if (!headless) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 8;
        app_ptr = std::make_unique<pez::render::WindowContextHandler>(app_title, window_size, settings, fullscreen ? sf::Style::Fullscreen : sf::Style::Default);
    }

    // Load resources
    pez::resources::registerFont("res/font.ttf", "font");
    pez::resources::registerTexture("res/circle.png", "circle");
    pez::resources::registerTexture("res/circle_empty.png", "circle_empty");
    pez::resources::registerTexture("res/shadow.png", "shadow");
    pez::resources::registerTexture("res/plant.png", "plant");
    pez::resources::registerTexture("res/agent.png", "agent");
    pez::resources::registerTexture("res/light.png", "light");

    // Register systems and entities
    pez::core::registerSingleton<SimulationConfiguration>(loader);
    pez::core::registerSingleton<SimulationState>();
    auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

    pez::core::registerEntity<Agent>();
    pez::core::registerEntity<Food>();
    pez::core::registerEntity<Plant>();

    pez::core::registerProcessor<PhysicSolver>(conf);
    pez::core::registerProcessor<PlantUpdater>();
    pez::core::registerProcessor<AgentPositionUpdater>();
    pez::core::registerProcessor<Simulation>(auto_restart);

    auto& simulation = pez::core::getProcessor<Simulation>();

    // Register events if app not in headless mode
    if (app_ptr) {
        pez::core::registerRenderer<Renderer>();
        auto renderer = &pez::core::getRenderer<Renderer>();

        auto reset_view = [=](){
            // Set focus to the center of the world
            pez::render::setFocus(conf.world_size_f * 0.5f);
            float const target_height = static_cast<float>(window_size.y) - 2.0f * consts::ui_margin;
            float const zoom = target_height / (conf.world_size_f.y + 2.0f * Renderer::background_margin);
            pez::render::setZoom(zoom);
        };

        pez::render::WindowContextHandler* app = app_ptr.get();
        app->getEventManager().addKeyPressedCallback(sf::Keyboard::S, [=](sfev::CstEv) {
            app->toggleUnlimitedFramerate();
            auto& state = pez::core::getSingleton<SimulationState>();
            state.full_speed = !state.full_speed;
        });
        app->getEventManager().addKeyPressedCallback(sf::Keyboard::E, [=](sfev::CstEv) {
            renderer->toggleRender();
        });
        app->getEventManager().addKeyPressedCallback(sf::Keyboard::Space, [=](sfev::CstEv) {
            auto& state = pez::core::getSingleton<SimulationState>();
            state.paused = !state.paused;
            pez::core::togglePause();
        });
        app->getEventManager().addKeyPressedCallback(sf::Keyboard::R, [=](sfev::CstEv) {
            reset_view();
        });
        app->getEventManager().addKeyPressedCallback(sf::Keyboard::G, [=](sfev::CstEv) {
            renderer->render_graphs = !renderer->render_graphs;
        });
        app->getEventManager().addMousePressedCallback(sf::Mouse::Right, [=](sfev::CstEv) {
            Vec2 const mouse_position = app->getWorldMousePosition();
            pez::core::getProcessor<Simulation>().selectAgent(mouse_position);
        });

        app->getEventManager().addKeyPressedCallback(sf::Keyboard::Escape, [=](sfev::CstEv) {
            if (renderer->exit_prompt.state != ExitPrompt::State::Hidden) {
                app->exit();
            } else {
                renderer->exit_prompt.show();
            }
        });

        reset_view();
    }

    RMean<float> update_time(consts::tick_rate);

    // Main loop
    float constexpr dt = 1.0f / static_cast<float>(consts::tick_rate);
    bool must_stop = false;
    while (!must_stop) {
        must_stop = app_ptr ? (!app_ptr->run() || (simulation.simulation_over && !auto_restart)) : (simulation.simulation_over && !auto_restart);

        sf::Clock clock;
        pez::core::update(dt);
        auto const ms = clock.getElapsedTime().asMilliseconds();
        update_time.addValue(static_cast<float>(ms));

        if (app_ptr) {
            pez::core::render({80, 80, 80});
            pez::core::getRenderer<Renderer>().time_state.frame_time = update_time.get();
        }
    }

    std::cout << "Simulation over: " << simulation.frame_count << std::endl;
    std::cout << "Exiting application" << std::endl;

    return 0;
}
