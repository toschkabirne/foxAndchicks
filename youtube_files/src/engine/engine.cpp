#include "engine.hpp"
#include "common/number_generator.hpp"


namespace pez::core
{

void createSystems(uint32_t thread_count_override)
{
    GlobalInstance::instance = new core::EngineInstance();
    // Create singletons provided by default by the engine
    createDefaultSingletons(thread_count_override);
}

void quit()
{
    GlobalInstance::instance->quit();
}

void render(sf::Color clear_color)
{
    pez::render::Context& context = *(GlobalInstance::instance->m_render_context);
    context.clear(clear_color);
    GlobalInstance::instance->m_entity_manager.render(context);
    context.display();
}

void update(float dt)
{
    GlobalInstance::instance->update(dt);
}

uint64_t getTick()
{
    return GlobalInstance::instance->tick;
}

float getTime()
{
    return GlobalInstance::instance->time;
}

float getWallClockTime()
{
    return GlobalInstance::instance->m_wall_clock.getElapsedTime().asSeconds();
}

void setPause(bool pause)
{
    GlobalInstance::instance->pause = pause;
}

void togglePause()
{
    GlobalInstance::instance->pause = !GlobalInstance::instance->pause;
}

bool isValidHandle(const Handle& handle)
{
    if (handle.id.class_id == EntityID::INVALID_ID) {
        return false;
    }
    return pez::core::GlobalInstance::instance->m_entity_manager.check_validity_callbacks[handle.id.class_id](handle);
}

Handle createEntityHandle(EntityID id)
{
    uint64_t const validity{
        GlobalInstance::instance->m_entity_manager.get_validity_callbacks[id.class_id](id.instance_id)
    };
    return {id.class_id, id.instance_id, validity};
}

bool isRunning()
{
    return !core::GlobalInstance::instance->pause;
}

void createDefaultSingletons(uint32_t thread_count_override)
{
    // Create RNG
    pez::core::registerSingleton<RealNumberGenerator<float>>();
    pez::core::getSingleton<RealNumberGenerator<float>>().setSeed(1);

    // The number of threads to use
    uint32_t thread_count;
    if (thread_count_override == 0) {
        thread_count = std::thread::hardware_concurrency();
    } else {
        thread_count = thread_count_override;
    }

    std::cout << "Using " << thread_count << " threads for multithreading." << std::endl;
    if (thread_count > 1) {
        pez::core::registerSingleton<tp::ThreadPool>(thread_count - 1);
    } else {
        pez::core::registerSingleton<tp::ThreadPool>(1);
    }
}
}
