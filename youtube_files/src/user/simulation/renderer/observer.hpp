#pragma once
#include <optional>

#include "engine/engine.hpp"
#include "engine/common/binary_io.hpp"

struct Observer
{
    struct Event
    {
        uint32_t tick = 0;
        Vec2     camera_position;
        float    zoom            = 1.0f;
        siv::ID  selected        = siv::InvalidID;
    };

    std::vector<Event> events;
    uint32_t           current_event = 0;

    bool over = false;

    /** Returns the event corresponding to the provided tick, if any
     *
     * @param tick The tick to search for
     * @return The associated event if any
     */
    std::optional<Event> getEvent(uint32_t tick)
    {
        if (current_event == events.size()) {
            if (!over) {
                over = true;
                std::cout << "Observer over" << std::endl;
            }
            return std::nullopt;
        }
        if (events[current_event].tick == tick) {
            return events[current_event++];
        }
        return std::nullopt;
    }

    void addEvent(uint32_t tick, Vec2 focus, float zoom)
    {
        events.push_back({
            tick,
            focus,
            zoom,
            siv::InvalidID
        });
    }

    void addEvent(uint32_t tick, siv::ID selected, float zoom)
    {
        events.push_back({
            tick,
            Vec2{},
            zoom,
            selected
        });
    }

    void writeToFile(std::string const& filename)
    {
        BinaryWriter writer(filename);
        writer.write(events.size());
        for (auto const& e : events) {
            writer.write(e);
        }
    }

    void loadFromFile(std::string const& filename)
    {
        BinaryReader reader(filename);
        auto const count = reader.read<size_t>();
        events.resize(count);
        for (size_t i{0}; i < count; ++i) {
            reader.readInto(events[i]);
        }
        current_event = 0;
    }


};
