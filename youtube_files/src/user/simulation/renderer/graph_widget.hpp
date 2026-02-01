#pragma once
#include <functional>
#include "engine/common/chart/line_chart.hpp"
#include "engine/common/card/card.hpp"
#include "engine/common/card/card_empty.hpp"
#include "engine/common/card/card_outlined.hpp"


struct GraphWidget : public sf::Transformable, public sf::Drawable
{
    float const background_radius = 20.0f;
    float const title_height      = 15.0f;
    Vec2 const  padding           = Vec2{background_radius, 1.5f * background_radius};
    float const outline           = 5.0f;
    Vec2  const outline_vec       = {outline, outline};
    sf::Color   color             = sf::Color::White;

    Vec2         size;
    Vec2         position;
    LineChart    chart;
    CardOutlined background;
    sf::Font*    font;
    sf::Text     title;
    mutable sf::Text     current_value;
    float        font_scale = 1.0f;

    mutable sf::VertexArray va_scale;
    sf::Color const scale_color  = {150, 150, 150};
    Vec2            inner_size;
    float           last_value = 0.0f;
    uint32_t        tick_x_period = 20;
    bool            new_data_added = true;

    std::function<std::string(uint32_t)> label_callback;
    std::function<std::string(float)>    current_value_callback;

    explicit
    GraphWidget(Vec2 size_, Vec2 position_ = {})
        : size{size_}
        , position{position_}
        , background{size, background_radius, outline, {50, 50, 50}}
        , chart{Vec2{}}
        , font{&pez::resources::getFont("font")}
        , va_scale{sf::PrimitiveType::Lines}
    {
        title.setString("Title");
        title.setFont(*font);
        title.setCharacterSize(24);
        title.setFillColor(sf::Color::White);
        title.setScale(font_scale, font_scale);

        current_value.setFont(*font);
        current_value.setCharacterSize(20);
        current_value.setFillColor(sf::Color::White);

        setPosition(position);
        chart.setPosition(padding + outline_vec + Vec2{0.0f, title_height});
        title.setPosition(Vec2{outline + 0.75f * padding.x, 0.35f * padding.y});

        label_callback = [](uint32_t i) {
            return toString(i);
        };

        current_value_callback = [](float x) {
            return toString(x);
        };

        setSize(size);
    }

    void setSize(Vec2 size_)
    {
        size = size_;
        inner_size = {size.x - 2.0f * (padding.x + outline), size.y - 2.0f * (padding.y + outline) - title_height};

        background.setOuterSize(size, outline);

        chart.setSize(inner_size);
        chart.setColor(color);
    }

    void setColor(sf::Color c)
    {
        color = c;
        chart.setColor(c);
        background.setOutlineColor(c);
    }

    void clear()
    {
        chart.clear();
    }

    void addValue(float v)
    {
        last_value = v;
        chart.addValue(v);
        new_data_added = true;
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();

        target.draw(background, states);

        // Render title
        target.draw(title, states);

        Vec2 const chart_area_start = padding + Vec2 {outline, outline + title_height};

        va_scale.clear();
        uint32_t const count = chart.values.getCount();
        float const    dx    = chart.size.x / static_cast<float>(count - 1);

        // X axis labels
        sf::Text scale_label;
        scale_label.setFont(*font);
        scale_label.setCharacterSize(18);
        scale_label.setFillColor(scale_color);
        scale_label.setScale(font_scale, font_scale);

        // Draw X ticks
        chart.values.foreach([&](uint32_t i, float v) {
            // Retrieve the idx of the current data point (not in the circular buffer, from the beginning)
            uint32_t const current_i = chart.getGlobalValueIndex(i);
            if (current_i % tick_x_period == 0) {
                // Current value position
                float const x = to<float>(i) * dx + chart_area_start.x;
                // Create vertical ticks
                sf::Vertex vertex{};
                vertex.color = scale_color;
                vertex.position = {x, chart_area_start.y}; // Top vertex
                va_scale.append(vertex);
                vertex.position = {x, chart_area_start.y + inner_size.y}; // Bottom vertex
                va_scale.append(vertex);

                // Add labels at the bottom
                scale_label.setString(label_callback(current_i));
                auto const bounds = scale_label.getLocalBounds();
                float const label_width = bounds.width;
                scale_label.setPosition(x - label_width * 0.5f, chart_area_start.y + inner_size.y);
                scale_label.setOrigin(bounds.left, 0.0f);
                target.draw(scale_label, states);
            }
        });

        // Draw Y ticks
        uint32_t const tick_count = 4;
        float const    y_span             = std::abs(chart.extremes.y - chart.extremes.x);
        float const    y_10_order         = std::floor(std::log10(y_span)) - 1.0f;
        // Could use fast pow but would break for negative power
        float const    tick_scale         = std::pow(10.0f, y_10_order);
        float const    tick_height_raw    = y_span / float(tick_count);
        float const    tick_height_scaled = std::ceil(tick_height_raw / tick_scale) * tick_scale;
        // Render all ticks
        for (uint32_t i{0}; i < tick_count; ++i) {
            sf::Vertex vertex{};
            vertex.color = scale_color;
            float const y_scale = chart.extremes.x + static_cast<float>(i) * tick_height_scaled;
            float const y = chart.getScaledY(y_scale) + chart_area_start.y;
            vertex.position = {chart_area_start.x, y};
            va_scale.append(vertex);
            vertex.position = {chart_area_start.x + inner_size.x, y};
            va_scale.append(vertex);

            float const label_margin = 4.0f;
            scale_label.setString(current_value_callback(y_scale));
            auto const bounds = scale_label.getLocalBounds();
            scale_label.setOrigin(bounds.width + label_margin, bounds.height + bounds.top + label_margin);
            scale_label.setPosition(chart_area_start.x + inner_size.x, y);
            target.draw(scale_label, states);
        }
        // Draw ticks
        target.draw(va_scale, states);

        // Draw chart
        target.draw(chart, states);

        // Draw current value on top
        if (chart.values.getCount()) {
            current_value.setScale(font_scale, font_scale);
            current_value.setString(current_value_callback(last_value));
            auto const bounds = current_value.getLocalBounds();
            current_value.setPosition(position.x + size.x - outline - background_radius - bounds.width, position.y - bounds.top + background_radius);
            target.draw(current_value, states);
        }
    }

    void setTitle(std::string const& title_)
    {
        title.setString(title_);
    }
};