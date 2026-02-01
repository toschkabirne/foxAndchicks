#pragma once

#include <SFML/Graphics.hpp>

#include "engine/common/card/card_outlined.hpp"


struct Gauge : public sf::Transformable, public sf::Drawable
{
    float padding           = 2.0f;
    float outline_thickness = 2.0f;
    float text_padding      = 20.0f;

    Vec2  size;
    Vec2  position;

    CardEmpty outline;
    Card      value;

    sf::Text title;
    sf::Font& font;

    Gauge()
        : outline{{}, 0.0f, sf::Color::White}
        , value{{}, 0.0f, sf::Color::White}
        , font{pez::resources::getFont("font")}
    {
        title.setFont(font);
        title.setCharacterSize(16);
        title.setFillColor({150, 150, 150});
        title.setString("Title");
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();
        target.draw(title, states);
        target.draw(value, states);
        target.draw(outline, states);
    }

    void setSize(Vec2 size_)
    {
        size = size_;
        outline = {size, size.y * 0.5f, sf::Color::White};
        outline.setThickness(outline_thickness);

        float const total_padding = outline_thickness + padding;
        value.setShape(outline.size - 2.0f * Vec2{total_padding, total_padding}, size.y * 0.5f - total_padding);
    }

    void setPosition(Vec2 position_)
    {
        position = position_;
        title.setPosition(position);
        auto const bounds = title.getGlobalBounds();
        float const title_height = bounds.height + text_padding;
        outline.setPosition(position + Vec2{0.0f, title_height});

        float const total_padding = outline_thickness + padding;
        value.setPosition(position + Vec2{total_padding, total_padding + title_height});
    }

    void setRatio(float ratio)
    {
        float const total_padding = outline_thickness + padding;
        value.setWidth(ratio * (outline.size.x - 2.0f * total_padding));
    }

    void setTitle(std::string const& string)
    {
        title.setString(string);
        setPosition(position);
    }
};
