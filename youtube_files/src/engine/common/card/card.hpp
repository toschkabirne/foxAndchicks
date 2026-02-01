#pragma once
#include "SFML/Graphics.hpp"
#include "../vec.hpp"
#include "utils.hpp"


struct Card : public sf::Drawable, public sf::Transformable
{
    sf::VertexArray va;
    sf::VertexArray va_shadow;

    Vec2            size;
    float           corner_radius = 0.0f;
    float           shadow_size   = 8.0f;
    sf::Color       color;

    uint32_t quality = 64;

    Card() = default;

    Card(Vec2 size_, float corner_radius_, sf::Color color_)
        : va{sf::PrimitiveType::TriangleFan}
        , va_shadow{sf::PrimitiveType::TriangleStrip}
        , size{size_}
        , corner_radius{corner_radius_}
        , color{color_}
    {
        updateGeometry();
    }

    void setColor(sf::Color color_)
    {
        color = color_;
        uint64_t const vertex_count = va.getVertexCount();
        for (uint64_t i{vertex_count}; i--;) {
            va[i].color = color;
        }
    }

    void setCornerRadius(float radius)
    {
        corner_radius = radius;
        updateGeometry();
    }

    void setShape(Vec2 size_, float corner_radius_)
    {
        corner_radius = corner_radius_;
        size = size_;
        updateGeometry();
    }

    void updateGeometry()
    {
        generateGeometry(va);
        generateGeometryShadow(va_shadow, {0, 0, 0, 30}, {0, 0, 0, 0});
    }

    void setShadowSize(float size_)
    {
        shadow_size = size_;
        updateGeometry();
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();
        if (shadow_size > 0.0f) {
            target.draw(va_shadow, states);
        }
        target.draw(va, states);
    }

    void setWidth(float width)
    {
        size.x = width;
        updateGeometry();
    }

    void setSize(Vec2 size_)
    {
        size = size_;
        updateGeometry();
    }

    void generateGeometry(sf::VertexArray& vertex_array)
    {
        float const radius = std::min(std::min(corner_radius, size.x * 0.5f), size.y * 0.5f);
        CardGeometryGenerator generator;
        generator.quality = quality;
        generator.size = size;
        generator.radius = radius;
        generator.start = 1;
        generator.color = color;

        uint64_t const vertex_count = generator.getVertexCount() + 1;
        vertex_array.resize(vertex_count);
        vertex_array[0].position = size * 0.5f;
        vertex_array[0].color = color;

        generator.generateVertex(&vertex_array);
    }

    void generateGeometryShadow(sf::VertexArray& vertex_array, sf::Color color_in, sf::Color color_out)
    {
        const float radius = std::min(std::min(corner_radius, size.x * 0.5f), size.y * 0.5f);

        CardGeometryGenerator generator;
        generator.quality = quality;
        vertex_array.resize(2 * generator.getVertexCount());

        generator.size = size;
        generator.radius = radius;
        generator.start = 0;
        generator.skip = 1;
        generator.color = color_in;
        generator.generateVertex(&vertex_array);

        generator.size = {size.x + 2.0f * shadow_size, size.y + 2.0f * shadow_size};
        generator.radius = radius + shadow_size;
        generator.start = 1;
        generator.skip = 1;
        generator.color = color_out;
        generator.generateVertex(&vertex_array, -Vec2{shadow_size, shadow_size});
    }
};

