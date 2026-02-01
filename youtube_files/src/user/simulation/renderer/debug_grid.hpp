#pragma once
#include "engine/engine.hpp"
#include "user/physics/physics.hpp"


struct DebugGrid
{
    PhysicSolver&   solver;
    sf::VertexArray va;

    DebugGrid()
        : solver{pez::core::getProcessor<PhysicSolver>()}
        , va{sf::PrimitiveType::Quads, 4 * to<uint32_t>(solver.grid.width * solver.grid.height)}
    {}

    void render(pez::render::Context& context)
    {
        uint32_t i{0};
        for (int32_t y{0}; y < solver.grid.height; ++y) {
            for (int32_t x{0}; x < solver.grid.width; ++x) {
                auto const xf = to<float>(x);
                auto const yf = to<float>(y);
                va[4 * i + 0].position = sf::Vector2f(xf    , yf);
                va[4 * i + 1].position = sf::Vector2f(xf + 1, yf);
                va[4 * i + 2].position = sf::Vector2f(xf + 1, yf + 1);
                va[4 * i + 3].position = sf::Vector2f(xf    , yf + 1);

                sf::Color const color = (solver.grid.get(x, y).objects_count > 0) ? sf::Color::Cyan : sf::Color::Black;
                va[4 * i + 0].color = color;
                va[4 * i + 1].color = color;
                va[4 * i + 2].color = color;
                va[4 * i + 3].color = color;
                ++i;
            }
        }

        context.draw(va);
    }
};
