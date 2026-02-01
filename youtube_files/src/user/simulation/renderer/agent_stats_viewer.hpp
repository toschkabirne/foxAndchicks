#pragma once
#include <SFML/Graphics.hpp>


struct AgentStatsViewer : public sf::Transformable, public sf::Drawable
{
    SimulationConfiguration const& conf;

    float stats_y_spacing{80.0f};
    float stats_padding{consts::widget_radius};
    Vec2  stats_size{280.0f, 5.0f * stats_y_spacing + 2.0f * stats_padding};
    Vec2  gauge_size{stats_size.x - 2.0f * stats_padding, 30.0f};

    CardOutlined stats_background;

    Gauge health;
    Gauge energy;
    Gauge split;
    Gauge reserve;

    sf::Text kill_count_text;
    sf::Text split_count_text;
    sf::Text kill_label;
    sf::Text split_label;

    explicit
    AgentStatsViewer(SimulationConfiguration const& conf_)
        : conf{conf_}
        , stats_background{stats_size, consts::widget_radius, consts::widget_outline, sf::Color::White}
    {
        stats_background.setFillColor(consts::widget_background_color);

        // Gauges
        sf::Color const gauge_title_color{200, 200, 200};
        uint32_t const gauge_title_size{32};
        health.title.setFillColor(gauge_title_color);
        health.title.setCharacterSize(gauge_title_size);
        health.value.setColor({42, 157, 143});
        health.setSize(gauge_size);
        health.setTitle("Health");

        energy.title.setFillColor(gauge_title_color);
        energy.title.setCharacterSize(gauge_title_size);
        energy.value.setColor({233, 196, 106});
        energy.setSize(gauge_size);
        energy.setTitle("Energy");

        split.title.setFillColor(gauge_title_color);
        split.title.setCharacterSize(gauge_title_size);
        split.value.setColor({231, 111, 81});
        split.setSize(gauge_size);
        split.setTitle("Split");

        reserve.title.setFillColor(gauge_title_color);
        reserve.title.setCharacterSize(gauge_title_size);
        reserve.value.setColor({231, 111, 81});
        reserve.setSize(gauge_size);
        reserve.setTitle("Reserve");

        // Text
        kill_count_text.setFont(pez::resources::getFont("font"));
        kill_count_text.setCharacterSize(32);
        kill_count_text.setFillColor(gauge_title_color);

        split_count_text.setFont(pez::resources::getFont("font"));
        split_count_text.setCharacterSize(32);
        split_count_text.setFillColor(gauge_title_color);

        kill_label.setFont(pez::resources::getFont("font"));
        kill_label.setCharacterSize(24);
        kill_label.setFillColor(gauge_title_color);

        split_label = kill_label;

        kill_label.setString("Kills");
        split_label.setString("Splits");

        // Set elements position
        updateLayout();
    }

    void updateLayout()
    {
        float current_y = consts::ui_margin;
        stats_background.setPosition(0.0f, current_y);

        float const gauge_x = stats_background.getPosition().x + stats_padding;

        current_y += stats_padding;

        reserve.setPosition({gauge_x, current_y});
        current_y += stats_y_spacing;
        health.setPosition({gauge_x, current_y});
        current_y += stats_y_spacing;
        energy.setPosition({gauge_x, current_y});
        current_y += stats_y_spacing;
        split.setPosition({gauge_x, current_y});
        current_y += 1.15f * stats_y_spacing;

        kill_label.setPosition(gauge_x, current_y);
        split_label.setPosition(stats_size.x * 0.5f, current_y);

        current_y += 0.35f * stats_y_spacing;

        kill_count_text.setPosition(gauge_x, current_y);
        kill_count_text.setFillColor(conf.pred_color);
        split_count_text.setPosition(stats_size.x * 0.5f, current_y);
        split_count_text.setFillColor(conf.prey_color);
    }

    void updateStats(Agent const& agent)
    {
        reserve.setRatio(agent.metabolism.reserve / conf.agent_reserve_max);
        health.setRatio(agent.getHealth() / conf.agent_health_max);
        energy.setRatio(agent.getEnergy() / conf.agent_energy_max);
        split.setRatio(agent.getSplitRatio());

        kill_count_text.setString(toString(agent.kill_count));
        split_count_text.setString(toString(agent.split_count));
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();

        target.draw(stats_background, states);

        target.draw(reserve, states);
        target.draw(health, states);
        target.draw(energy, states);
        target.draw(split, states);

        target.draw(kill_label, states);
        target.draw(split_label, states);

        target.draw(kill_count_text, states);
        target.draw(split_count_text, states);
    }
};