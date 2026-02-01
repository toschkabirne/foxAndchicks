#pragma once
#include "engine/common/card/card_outlined.hpp"


struct Minimap : public sf::Transformable, public sf::Drawable
{
    Vec2  const  size;
    CardOutlined background;

    sf::VertexArray const* va_agents = nullptr;
    sf::VertexArray const* va_food   = nullptr;
    sf::VertexArray const* va_plants = nullptr;

    explicit
    Minimap(Vec2 size_, sf::VertexArray const& va_agents_, sf::VertexArray const& va_food_, sf::VertexArray const& va_plants_)
        : size{size_}
        , background{size, consts::widget_radius, consts::widget_outline, sf::Color::White}
        , va_agents{&va_agents_}
        , va_food{&va_food_}
        , va_plants{&va_plants_}
    {
        background.setFillColor(consts::widget_background_color);
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();

        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();

        target.draw(background, states);

        Vec2 const  world_size_f{conf.world_size_f};
        Vec2 const  padding{consts::widget_radius, consts::widget_radius};
        float const scale{(size.x - 2.0f * padding.x) / world_size_f.x};
        Vec2 const offset{background.getPosition() + padding};

        sf::Transform transform;
        transform.translate(offset);
        transform.scale(scale, scale);
        states.transform *= transform;

        if (va_food) {
            target.draw(*va_food  , states);
        }
        if (va_plants) {
            target.draw(*va_plants, states);
        }
        if (va_agents) {
            target.draw(*va_agents, states);
        }

        Vec2 const render_size    = pez::render::getRenderSize();
        Vec2 const camera_pos     = pez::render::getFocus();
        Vec2 const viewport_size  = render_size / pez::render::getZoom();

        auto const clamp = [&](Vec2 v) -> Vec2 {
            return {std::min(std::max(v.x, 0.0f), world_size_f.x),
                    std::min(std::max(v.y, 0.0f), world_size_f.y)};
        };

        Vec2 const top_left_position = {
            clamp(camera_pos - viewport_size * 0.5f)
        };

        Vec2 const bot_right_position = {
            clamp(camera_pos + viewport_size * 0.5f)
        };

        Vec2 const viewport_map_size = (bot_right_position - top_left_position);

        if (viewport_map_size.x > 0.0f && viewport_map_size.y > 0.0f) {
            if ((top_left_position.x > 0.0f) || (bot_right_position.x < world_size_f.x) ||
                (top_left_position.y > 0.0f) || (bot_right_position.y < world_size_f.y) ) {
                sf::RectangleShape viewport_map{viewport_map_size};
                viewport_map.setPosition(top_left_position);
                viewport_map.setOutlineThickness(2.0f);
                viewport_map.setOutlineColor({200, 200, 200});
                viewport_map.setFillColor(sf::Color::Transparent);
                target.draw(viewport_map, states);
            }
        }
    }
};
