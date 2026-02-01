#pragma once
#include <cstdint>

#include "engine/core/instance.hpp"
#include "engine/core/timer.hpp"
#include "engine/render/render.hpp"
#include "engine/resources/resources.hpp"

#include "engine/common/thread_pool/thread_pool.hpp"


namespace pez::core
{

void     createSystems(uint32_t thread_count_override);
void     quit();
void     update(float dt);
void     render(sf::Color clear_color = sf::Color::Black);
uint64_t getTick();
float    getTime();
float    getWallClockTime();
void     setPause(bool pause);
void     togglePause();
bool     isRunning();

Handle createEntityHandle(EntityID id);

void createDefaultSingletons(uint32_t thread_count_override);

template<typename T>
uint32_t getClassID()
{
    return EntityContainer<T>::class_id;
}

template<typename TEntity>
Handle createEntityHandle(ID instance_id)
{
    uint32_t const class_id = getClassID<TEntity>();
    uint64_t const validity{
        GlobalInstance::instance->m_entity_manager.get_validity_callbacks[class_id](instance_id)
    };
    return {class_id, instance_id, validity};
}

template<typename T>
static siv::Vector<T>& getData()
{
    return core::EntityContainer<T>::data;
}

template<typename T, typename... Arg>
static ID create(Arg&&... args)
{
    return core::EntityContainer<T>::create(std::forward<Arg>(args)...);
}

template<typename T, typename... Arg>
static void createMultiple(uint32_t count, Arg&&... args)
{
    for (uint32_t i{count}; i--;) {
        core::EntityContainer<T>::create(std::forward<Arg>(args)...);
    }
}

template<typename T>
static T& get(siv::ID id)
{
    return core::EntityContainer<T>::data[id];
}

template<typename T>
siv::Handle<T> getHandle(siv::ID id)
{
    return core::EntityContainer<T>::data.createHandle(id);
}

template<typename T>
T& getProcessor()
{
    return *core::System<T>::instance;
}

template<typename T>
T& getRenderer()
{
    return *core::System<T>::instance;
}

template<typename T>
T& getSingleton()
{
    return *core::Singleton<T>::instance;
}

template<typename T, typename... Arg>
siv::Handle<T> createGetHandle(Arg&&... args)
{
    const siv::ID id = core::EntityContainer<T>::create(std::forward<Arg>(args)...);
    return core::EntityContainer<T>::data.createHandle(id);
}

template<typename T>
bool isValid(const core::Handle& handle)
{
    return GlobalInstance::instance->m_entity_manager.isValid<T>(handle);
}

bool isValidHandle(const core::Handle& handle);

template<typename T>
T& get(const core::Handle& handle)
{
    return get<T>(handle.id.instance_id);
}

template<typename T>
bool isInstanceOf(const core::Handle& handle)
{
    return core::EntityContainer<T>::class_id == handle.id.class_id;
}

template<typename T>
bool isInstanceOf(const core::EntityID& id)
{
    return core::EntityContainer<T>::class_id == id.class_id;
}

template<typename T>
void registerEntity()
{
    core::GlobalInstance::instance->m_entity_manager.registerEntity<T>();
}

template<typename T>
void registerDataEntity()
{
    core::GlobalInstance::instance->m_entity_manager.registerDataEntity<T>();
}

template<typename T, typename... TArg>
void registerProcessor(TArg&&... args)
{
    core::GlobalInstance::instance->m_entity_manager.registerProcessor<T>(std::forward<TArg>(args)...);
}

template<typename T>
bool isRegistered()
{
    return System<T>::isRegistered();
}

template<typename T, typename... TArg>
static void registerRenderer(TArg&&... args)
{
    core::GlobalInstance::instance->m_entity_manager.registerRenderer<T>(std::forward<TArg>(args)...);
}

template<typename T, typename... TArg>
static void registerSingleton(TArg&&... args)
{
    core::GlobalInstance::instance->m_entity_manager.registerSingleton<T>(std::forward<TArg>(args)...);
}

template<typename T>
void remove(ID id)
{
    EntityContainer<T>::data.erase(id);
}

template<typename T, typename TCallback>
void foreach(TCallback&& callback) {
    static_assert(std::is_convertible<T*, Entity*>::value, "Can only iterate on Entity derived objects");
    std::vector<T>& data = core::EntityContainer<T>::data.getData();
    const uint64_t count = core::EntityContainer<T>::data.size();
    for (uint64_t i{0}; i<count; ++i) {
        if (!data[i].isRemoved()) {
            callback(data[i]);
        }
    }
}

template<typename T, typename TCallback>
void foreachAbort(TCallback&& callback) {
    static_assert(std::is_convertible<T*, Entity*>::value, "Can only iterate on Entity derived objects");
    std::vector<T>& data = core::EntityContainer<T>::data.getData();
    const uint64_t count = core::EntityContainer<T>::data.size();
    for (uint64_t i{0}; i<count; ++i) {
        if (!data[i].isRemoved()) {
            if (callback(data[i])) {
                return;
            }
        }
    }
}

template<typename T, typename TCallback>
void parallelForeach(TCallback&& callback) {
    static_assert(std::is_convertible<T*, Entity*>::value, "Can only iterate on Entity derived objects");
    std::vector<T>& data  = core::EntityContainer<T>::data.getData();
    auto const      count = static_cast<uint32_t>(core::EntityContainer<T>::data.size());

    auto& tp = pez::core::getSingleton<tp::ThreadPool>();
    tp.dispatch(count, [&data, callback](uint32_t start, uint32_t end) {
        for (uint32_t i{start}; i < end; ++i) {
            if (!data[i].isRemoved()) {
                callback(data[i]);
            }
        }
    });
}

template<typename T, typename TCallback>
void parallelForeachEnumerate(TCallback&& callback) {
    static_assert(std::is_convertible<T*, Entity*>::value, "Can only iterate on Entity derived objects");
    std::vector<T>& data  = core::EntityContainer<T>::data.getData();
    auto const      count = static_cast<uint32_t>(core::EntityContainer<T>::data.size());

    auto& tp = pez::core::getSingleton<tp::ThreadPool>();
    tp.dispatch(count, [&data, callback](uint32_t start, uint32_t end) {
        for (uint32_t i{start}; i < end; ++i) {
            if (!data[i].isRemoved()) {
                callback(i, data[i]);
            }
        }
    });
}

}
