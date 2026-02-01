#pragma once
#include "engine/common/vec.hpp"
#include "engine/common/math.hpp"


namespace consts
{
    // Simulation
    uint32_t constexpr tick_rate = 60;

    // UI
    float constexpr ui_margin = 20.0f;
    float constexpr widget_radius = 20.0f;
    float constexpr widget_outline = 5.0f;
    sf::Color const widget_background_color{50, 50, 50, 230};
}