#pragma once
#include <SFML/Graphics.hpp>

sf::Image getImageFromLogo(uint32_t width, uint32_t height, uint32_t byte_per_pixel, unsigned char const* data)
{
    sf::Image image;
    image.create(width, height);

    for (uint32_t y{0}; y < height; ++y) {
        for (uint32_t x{0}; x < width; ++x) {
            uint32_t const idx{(y * width + x) * byte_per_pixel};
            uint8_t pxl[4] = {255, 255, 255, 255};
            for (uint32_t i{0}; i < byte_per_pixel; ++i) {
                pxl[i] = data[idx + i];
            }
            // Will not work nicely with depth < 3
            image.setPixel(x, y, {pxl[0], pxl[1], pxl[2], pxl[3]});
        }
    }

    return image;
}