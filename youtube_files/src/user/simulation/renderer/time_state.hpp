#pragma once
#include "engine/common/card/card_outlined.hpp"
#include "user/simulation/simulation.hpp"
#include "user/simulation/simulation_state.hpp"


struct TimeState : public sf::Transformable, public sf::Drawable
{
    float const padding = 10.0f;
    Vec2 const  size = {400.0f, 160.0f};
    Vec2 const  button_size = {(size.x - 4.0f * padding - 2.0f * consts::widget_outline) / 3.0f, 45.0f};

    CardOutlined background;

    sf::Text sim_time_label_text;
    sf::Text sim_time_text;
    sf::Text tick_count_label_text;
    sf::Text tick_count_text;
    sf::Text frame_time_label_text;
    sf::Text frame_time_text;

    Card pause;
    Card play;
    Card full_speed;

    sf::Text pause_text;
    sf::Text play_text;
    sf::Text full_speed_text;

    float frame_time = 0.0f;

    TimeState()
        : background{size, consts::widget_radius, consts::widget_outline, sf::Color::White}
        , pause{button_size, 10.0f, sf::Color::White}
        , play{button_size, 10.0f, sf::Color::White}
        , full_speed{button_size, 10.0f, sf::Color::White}
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        float const ui_scale = conf.ui_scale;
        Vec2 const render_size{pez::render::getRenderSize()};
        Vec2 const outline_size{background.getOutlineSize()};

        float const x = (render_size.x - outline_size.x * ui_scale) * 0.5f;
        float const y = (render_size.y - (outline_size.y + consts::ui_margin) * ui_scale);
        setPosition(x, y);

        background.setFillColor(consts::widget_background_color);

        Vec2 const offset = -Vec2{consts::widget_outline, consts::widget_outline};

        sf::Color const text_color{200, 200, 200};
        sf::Font const& font = pez::resources::getFont("font");

        float const secondary_offset_y = 15.0f;

        // Time
        sim_time_label_text.setFont(font);
        sim_time_label_text.setCharacterSize(32);
        sim_time_label_text.setFillColor(text_color);
        sim_time_label_text.setString("Time");
        auto const time_label_bounds = sim_time_label_text.getGlobalBounds();
        sim_time_label_text.setOrigin(time_label_bounds.width * 0.5f, 0.0f);
        sim_time_label_text.setPosition({offset.x + size.x * 0.5f, offset.y + padding});

        sim_time_text.setFont(font);
        sim_time_text.setCharacterSize(48);
        sim_time_text.setPosition(offset.x + size.x * 0.5f, offset.y + time_label_bounds.height + 2.0f * padding);

        // Ticks
        float const tick_x = offset.x + size.x * 0.15f;
        tick_count_label_text.setFont(font);
        tick_count_label_text.setCharacterSize(20);
        tick_count_label_text.setFillColor(text_color);
        tick_count_label_text.setString("Ticks");
        auto const ticks_label_bounds = tick_count_label_text.getGlobalBounds();
        tick_count_label_text.setOrigin(ticks_label_bounds.width * 0.5f, 0.0f);
        tick_count_label_text.setPosition({tick_x, offset.y + padding + secondary_offset_y});

        tick_count_text.setFont(font);
        tick_count_text.setCharacterSize(28);
        tick_count_text.setPosition(tick_x, offset.y + time_label_bounds.height + 2.0f * padding + secondary_offset_y);

        // Frame
        float const frame_x = offset.x + size.x * 0.85f;
        frame_time_label_text.setFont(font);
        frame_time_label_text.setCharacterSize(20);
        frame_time_label_text.setFillColor(text_color);
        frame_time_label_text.setString("Frame");
        auto const frame_time_label_bounds = frame_time_label_text.getGlobalBounds();
        frame_time_label_text.setOrigin(frame_time_label_bounds.width * 0.5f, 0.0f);
        frame_time_label_text.setPosition({frame_x, offset.y + padding + secondary_offset_y});

        frame_time_text.setFont(font);
        frame_time_text.setCharacterSize(28);
        frame_time_text.setPosition(frame_x, offset.y + time_label_bounds.height + 2.0f * padding + secondary_offset_y);

        // Buttons
        float const buttons_x = offset.x + 2.0f * consts::widget_outline + padding;
        float const buttons_y = offset.y + size.y - padding - button_size.y;
        pause.setPosition(buttons_x, buttons_y);
        play.setPosition(buttons_x + button_size.x + padding, buttons_y);
        full_speed.setPosition(buttons_x + 2.0f * (button_size.x + padding), buttons_y);

        // Buttons text
        uint32_t const text_size = 18;
        pause_text.setFont(font);
        pause_text.setCharacterSize(text_size);
        pause_text.setFillColor(sf::Color::Black);
        pause_text.setString("pausd"); // Add an additional "d" to match others height, I am not proud
        auto const pause_bounds = pause_text.getLocalBounds();
        pause_text.setString("pause");
        pause_text.setOrigin(pause_bounds.width * 0.5f + pause_bounds.left, pause_bounds.height * 0.5f + pause_bounds.top);
        pause_text.setPosition(pause.getPosition() + pause.size * 0.5f);

        play_text.setFont(font);
        play_text.setCharacterSize(text_size);
        play_text.setFillColor(sf::Color::Black);
        play_text.setString("play");
        auto const play_bounds = play_text.getLocalBounds();
        play_text.setOrigin(play_bounds.width * 0.5f + play_bounds.left, play_bounds.height * 0.5f + play_bounds.top);
        play_text.setPosition(play.getPosition() + play.size * 0.5f);

        full_speed_text.setFont(font);
        full_speed_text.setCharacterSize(text_size);
        full_speed_text.setFillColor(sf::Color::Black);
        full_speed_text.setString("max speed");
        auto const full_speed_bounds = full_speed_text.getLocalBounds();
        full_speed_text.setOrigin(full_speed_bounds.width * 0.5f + full_speed_bounds.left, full_speed_bounds.height * 0.5f + full_speed_bounds.top);
        full_speed_text.setPosition(full_speed.getPosition() + full_speed.size * 0.5f);
    }

    void update()
    {
        auto const& simulation = pez::core::getProcessor<Simulation>();
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        float constexpr dt = 1.0f / static_cast<float>(consts::tick_rate);

        // Time
        sim_time_text.setString(toString(static_cast<float>(simulation.frame_count) * dt, 0) + "s");
        sim_time_text.setOrigin(sim_time_text.getGlobalBounds().width * 0.5f, 0.0f);

        // Ticks
        tick_count_text.setString(toString(simulation.frame_count));
        tick_count_text.setOrigin(tick_count_text.getGlobalBounds().width * 0.5f, 0.0f);

        // Frame time
        frame_time_text.setString(toString(frame_time, 1) + "ms");
        frame_time_text.setOrigin(frame_time_text.getGlobalBounds().width * 0.5f, 0.0f);

        // Play / Pause
        sf::Color const disabled_color = {120, 120, 120};
        if (pez::core::isRunning()) {
            play.setColor(conf.plant_color);
            pause.setColor(disabled_color);
        } else {
            play.setColor(disabled_color);
            pause.setColor(conf.food_color);
        }

        // Speed
        if (pez::core::getSingleton<SimulationState>().full_speed) {
            full_speed.setColor(conf.prey_color);
        } else {
            full_speed.setColor(disabled_color);
        }
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();

        target.draw(background, states);

        // Time
        target.draw(sim_time_label_text, states);
        target.draw(sim_time_text, states);

        // Ticks
        target.draw(tick_count_label_text, states);
        target.draw(tick_count_text, states);

        // Frame
        target.draw(frame_time_label_text, states);
        target.draw(frame_time_text, states);

        // Buttons
        target.draw(pause, states);
        target.draw(play, states);
        target.draw(full_speed, states);
        target.draw(pause_text, states);
        target.draw(play_text, states);
        target.draw(full_speed_text, states);
    }
};
