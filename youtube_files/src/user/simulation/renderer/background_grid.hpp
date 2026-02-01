#pragma once

struct BackgroundGrid
{
    sf::VertexArray va;

    float small_width = 0.05f;
    float large_width = 0.2f;

    BackgroundGrid(IVec2 size, uint32_t small_tick, uint32_t large_tick)
        : va{sf::PrimitiveType::Quads}
    {
        uint32_t const vertical_small_count  {size.x / small_tick};
        uint32_t const horizontal_small_count{size.y / small_tick};

        uint32_t const vertical_large_count  {size.x / large_tick};
        uint32_t const horizontal_large_count{size.y / large_tick};

        va.resize((vertical_small_count + horizontal_small_count + vertical_large_count + horizontal_large_count + 4) * 4);

        float half_small{small_width * 0.5f};
        // Vertical small
        for (uint32_t i{1}; i < vertical_small_count; ++i) {
            uint32_t const idx{i * 4};
            auto const x = to<float>(i * small_tick);
            va[idx + 0].position = {x - half_small, 0.0f};
            va[idx + 1].position = {x + half_small, 0.0f};
            va[idx + 2].position = {x + half_small, to<float>(size.y)};
            va[idx + 3].position = {x - half_small, to<float>(size.y)};
        }
        uint32_t global_index{vertical_small_count - 1};

        // Horizontal small
        for (uint32_t i{1}; i < horizontal_small_count; ++i) {
            uint32_t const idx{(i + global_index) * 4};
            auto const y = to<float>(i * small_tick);
            va[idx + 0].position = {0.0f             , y - half_small};
            va[idx + 1].position = {0.0f             , y + half_small};
            va[idx + 2].position = {to<float>(size.x), y + half_small};
            va[idx + 3].position = {to<float>(size.x), y - half_small};
        }
        global_index += horizontal_small_count - 1;

        float const half_large{large_width * 0.5f};
        // Vertical large
        for (uint32_t i{0}; i < vertical_large_count + 1; ++i) {
            uint32_t const idx{(i + global_index) * 4};
            auto const x = to<float>(i * large_tick);
            va[idx + 0].position = {x - half_large, -half_large};
            va[idx + 1].position = {x + half_large, -half_large};
            va[idx + 2].position = {x + half_large, to<float>(size.y) + half_large};
            va[idx + 3].position = {x - half_large, to<float>(size.y) + half_large};
        }
        global_index += vertical_large_count + 1;

        // Horizontal large
        for (uint32_t i{0}; i < horizontal_large_count + 1; ++i) {
            uint32_t const idx{(i + global_index) * 4};
            auto const y = to<float>(i * large_tick);
            va[idx + 0].position = {0.0f             , y - half_large};
            va[idx + 1].position = {0.0f             , y + half_large};
            va[idx + 2].position = {to<float>(size.x), y + half_large};
            va[idx + 3].position = {to<float>(size.x), y - half_large};
        }
    }

    void setColor(sf::Color color)
    {
        size_t const vertex_count = va.getVertexCount();
        for (size_t i{vertex_count}; i--;) {
            va[i].color = color;
        }
    }

    void render(pez::render::Context& context)
    {
        context.draw(va);
    }
};