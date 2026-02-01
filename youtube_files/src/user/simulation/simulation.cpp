#include "simulation.hpp"
#include "renderer/renderer.hpp"


void Simulation::restart()
{
    auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

    if (seed && frame_count) {
        std::cout << "Simulation over, seed: " << seed-1 << ", frames: " << frame_count << std::endl;
    }
    std::cout << "Starting new simulation, seed: " << seed << std::endl;
    rng.setSeed(seed);
    ++seed;

    // RNG fingerprint
    std::cout << "RNG Fingerprint [";
    for (uint32_t i{10}; i--;) {
        std::cout << static_cast<uint32_t>(rng.getUnder(100.0f));
        if (i > 0) {
            std::cout << "|";
        }
    }
    std::cout << "]" << std::endl;

    frame_count = 0;

    // Clear everything
    pez::core::getData<Plant>().clear();
    pez::core::getData<Agent>().clear();
    pez::core::getData<Food>().clear();
    solver.clear();

    auto& plant_updater = pez::core::getProcessor<PlantUpdater>();
    plant_updater.clear();
    plant_updater.createInitialEnv();

    createInitialPopulation();

    // Reset selected agent to avoid invalid handle when simulation is restarted
    selected = {};
    if (pez::core::isRegistered<Renderer>()) {
        pez::core::getRenderer<Renderer>().agent_viewer.selected_last = {};
    }
}
