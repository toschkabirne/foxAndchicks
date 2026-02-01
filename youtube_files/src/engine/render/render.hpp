#pragma once
#include "engine/common/vec.hpp"


namespace pez::render
{
    void setFocus(Vec2 focus);
    void setZoom(float zoom);
    void clear(sf::Color color);
    Vec2 getRenderSize();

    [[nodiscard]]
    Vec2  getFocus();
    [[nodiscard]]
    float getZoom();

    void moveFocusRelative(Vec2 d);
}