#pragma once

#pragma once
#include "SFML/Graphics.hpp"
#include "../vec.hpp"
#include "utils.hpp"
#include "./card.hpp"
#include "./card_empty.hpp"

struct CardOutlined : public sf::Drawable, public sf::Transformable
{
    Card      background;
    CardEmpty outline;

    CardOutlined(Vec2 size_, float corner_radius_, float thickness, sf::Color color)
        : background{size_, corner_radius_, color}
        , outline{size_, corner_radius_ + thickness, sf::Color::White}
    {
        setOuterSize(size_, thickness);
    }

    void setOuterSize(Vec2 size, float thickness)
    {
        background.setSize(size - 2.0f * Vec2{thickness, thickness});
        setOutlineThickness(thickness);
    }

    void setInnerSize(Vec2 size, float thickness)
    {
        background.setSize(size);
        setOutlineThickness(thickness);
    }

    void setFillColor(sf::Color color)
    {
        background.setColor(color);
    }

    void setOutlineColor(sf::Color color)
    {
        outline.setColor(color);
    }

    /// Sets the thickness of the outline outside of the background (global size = inner_size + thickness)
    void setOutlineThickness(float thickness)
    {
        Vec2 const base_size{background.size};
        outline.setThickness(thickness);
        outline.setSize(base_size + 2.0f * Vec2{thickness, thickness});
    }

    void setOutlineShadowSize(float size_)
    {
        outline.setShadowSize(size_);
    }

    [[nodiscard]]
    float getThickness() const
    {
        return outline.thickness;
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();
        target.draw(outline, states);
        states.transform.translate(outline.thickness, outline.thickness);
        target.draw(background, states);
    }

    [[nodiscard]]
    Vec2 getOutlineSize() const
    {
        return outline.size;
    }
};

