#pragma once
#include <SFML/Graphics.hpp>


struct VertexArrayUtils
{
    static void createQuad(sf::VertexArray& va, uint32_t index, sf::Vector2f position, float size, sf::Color color)
    {
        va[index + 0].position = position + sf::Vector2f{-size, -size};
        va[index + 1].position = position + sf::Vector2f{ size, -size};
        va[index + 2].position = position + sf::Vector2f{ size,  size};
        va[index + 3].position = position + sf::Vector2f{-size,  size};
        va[index + 0].color = color;
        va[index + 1].color = color;
        va[index + 2].color = color;
        va[index + 3].color = color;
    }

    static void createTexturedQuad(sf::VertexArray& va, uint32_t index, Vec2 position, float size, sf::Color color, float texture_size = 1.0f)
    {
        createQuad(va, index, position, size, color);
        va[index + 0].texCoords = {0.0f        , 0.0f};
        va[index + 1].texCoords = {texture_size, 0.0f};
        va[index + 2].texCoords = {texture_size, texture_size};
        va[index + 3].texCoords = {0.0f        , texture_size};
    }

    static void createRect(sf::VertexArray& va, uint32_t index, Vec2 position, Vec2 size)
    {
        va[index + 0].position = position - size;
        va[index + 1].position = position + Vec2{ size.x, -size.y};
        va[index + 2].position = position + size;
        va[index + 3].position = position + Vec2{-size.x,  size.y};
    }
};