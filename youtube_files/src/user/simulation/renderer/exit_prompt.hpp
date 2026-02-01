#pragma once
#include <SFML/Graphics.hpp>
#include "engine/common/card/card_outlined.hpp"
#include "engine/common/interpolation/interpolated_value.hpp"


struct ExitPrompt final : public sf::Drawable, public sf::Transformable
{
    static constexpr float timeout = 2.0f;

    Vec2 const size = {450.0f, 100.0f};
    float const time_bar_height = 8.0f;

    float const hide_y = -size.y - consts::ui_margin;
    float const show_y = consts::ui_margin;

    pez::InterpolatedFloat current_y;
    Vec2  position;
    float time = 0.0f;

    enum class State {
        Hidden,
        Shown,
        MovingDown,
        MovingUp,
    };

    State state = State::Hidden;

    CardOutlined background;
    Card         time_bar{{}, 0.0f, sf::Color{6, 214, 160}};
    Card         time_bar_background{{}, 0.0f, sf::Color{20, 20, 20}};

    sf::Clock update_clock;

    sf::Text text;

    explicit
    ExitPrompt()
        : background{size, consts::widget_radius, consts::widget_outline, consts::widget_background_color}
    {
        background.setOutlineColor({231, 111, 81});

        text.setFont(pez::resources::getFont("font"));
        text.setFillColor(sf::Color::White);
        text.setCharacterSize(28);
        text.setString("Press Esc. again to exit");

        auto const bounds = text.getGlobalBounds();
        text.setOrigin(bounds.left, 0.0f);
        text.setPosition((size.x - bounds.width) * 0.5f, consts::ui_margin);

        current_y.setInterpolation(pez::InterpolationFunction::EaseInOutQuint);
        current_y.setSpeed(2.0f);
        current_y.setValueInstant(hide_y);

        time_bar.setPosition(text.getPosition() + Vec2{0.0f, consts::ui_margin + bounds.height});
        time_bar.setSize({bounds.width, time_bar_height});
        time_bar.setShadowSize(0.0f);

        time_bar_background.setPosition(time_bar.getPosition());
        time_bar_background.setSize(time_bar.size);
    }

    void setX(float x)
    {
        sf::Transformable::setPosition(x, hide_y);
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        if (state == State::Hidden) {
            return;
        }
        sf::Transform tf = getTransform();
        tf.translate(0.0f, current_y);
        states.transform *= tf;
        target.draw(background, states);
        target.draw(text, states);

        target.draw(time_bar_background, states);
        target.draw(time_bar, states);
    }

    void update()
    {
        if (current_y.isDone()) {
            if (state == State::MovingDown) {
                state = State::Shown;
                update_clock.restart();
            } else if (state == State::MovingUp) {
                state = State::Hidden;
            } else if (state == State::Shown) {
                time = update_clock.getElapsedTime().asSeconds();
                if (time > timeout) {
                    hide();
                }
            }
        }

        float const time_ratio = std::max(0.0f, 1.0f - time / timeout);
        time_bar.setScale({time_ratio, 1.0f});
    }

    void hide()
    {
        current_y = hide_y;
        state = State::MovingUp;
    }

    void show()
    {
        state = State::MovingDown;
        current_y = show_y;
        time = 0.0f;
    }
};