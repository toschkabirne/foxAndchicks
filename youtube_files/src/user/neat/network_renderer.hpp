#pragma once

#include <cassert>
#include <vector>

#include <SFML/Graphics.hpp>

#include "engine/common/utils.hpp"
#include "engine/common/render/utils.hpp"
#include "engine/common/smooth/smooth_value.hpp"
#include "user/neat/network.hpp"
#include "engine/common/card/card_outlined.hpp"


struct NetworkRenderer : public sf::Drawable, sf::Transformable
{
    struct DrawableNode
    {
        sf::Vector2f position;
        uint32_t     layer    = 0;
        SmoothFloat  value;
        uint32_t     depth = 0;

        DrawableNode()
        {
            value.setSpeed(10.0f);
            value.setInterpolationFunction(Interpolation::Linear);
            value.setValueInstant(0.0f);
        }
    };

    struct DrawableConnection
    {
        sf::Vector2f start;
        sf::Vector2f end;
        SmoothFloat  width;

        DrawableConnection()
        {
            width.setSpeed(5.0f);
            width.setInterpolationFunction(Interpolation::Linear);
        }
    };

    struct NetworkArchitecture
    {
        std::vector<uint32_t> layers;
        uint32_t              max_layer_size = 0;
        uint32_t              hidden_count = 0;
        uint32_t              connection_count = 0;

        void reset()
        {
            layers.clear();
            hidden_count = 0;
            connection_count = 0;
        }
    };

    Vec2  node_spacing = {16.0f, 16.0f};
    float node_radius  = 9.0f;

    NetworkArchitecture architecture;

    mutable sf::Text         text_label;
    std::vector<std::string> labels;
    float                    label_offset = 0.0f;

    std::vector<DrawableNode>       nodes;
    std::vector<DrawableConnection> connections;

    Vec2 size     = {};
    Vec2 position = {};

    sf::VertexArray connections_va;

    float background_outline_thickness = 5.0f;
    float background_radius            = 20.0f;

    CardOutlined background;

    bool disable_info = false;

    NetworkRenderer()
        : background({}, background_radius, background_outline_thickness, sf::Color::White)
    {
        background.setOutlineColor(sf::Color::White);
        background.setFillColor(consts::widget_background_color);

        setFont(pez::resources::getFont("font"));
    }

    void setFont(sf::Font& font)
    {
        text_label.setFont(font);
        text_label.setCharacterSize(16);
        text_label.setFillColor(sf::Color::White);
    }

    void initialize(nt::Network const& nw)
    {
        architecture.reset();
        nodes.clear();
        connections.clear();

        architecture.hidden_count     = nw.info.hidden;
        architecture.connection_count = nw.connection_count;

        label_offset = 0.0f;
        for (auto const& l : labels) {
            text_label.setString(l);
            auto const bounds = text_label.getGlobalBounds();
            label_offset = std::max(label_offset, bounds.width + background_radius * 0.5f);
        }

        // Create nodes
        float const total_padding = background_radius;
        architecture.layers.resize(getMaxDepth(nw) + 1, 0);
        size.x = static_cast<float>(getMaxDepth(nw) + 1) * (node_radius * 2.0f + node_spacing.x) - node_spacing.x + node_radius * 0.5f + 2.0f + label_offset + 2.0f * total_padding;

        nodes.resize(nw.info.getNodeCount());
        for (uint32_t i{0}; i < nw.info.getNodeCount(); ++i) {
            auto&       node = nodes[i];
            auto const& n    = nw.slots[i].node;
            node.layer       = n.depth;
            /* Little hack to ensure the output layer is last even with no connections.
             * A better solution would be to do this directly when building the network. */
            if (i >= nw.info.inputs && i < nw.info.inputs + nw.info.outputs && n.depth == 0) {
                node.layer = 1;
            }
            node.position.x = total_padding + label_offset + to<float>(node.layer) * (node_radius * 2.0f + node_spacing.x);
            node.position.y = total_padding + to<float>(architecture.layers[node.layer]) * (node_radius * 2.0f + (i ? node_spacing.y : 0.0f));
            ++architecture.layers[node.layer];
        }

        // Center layers
        auto const layer_size = getMax<uint32_t>(architecture.layers, [](uint32_t x) { return x; });
        float const max_layer_height = getLayerHeight(layer_size) + 4.0f;
        architecture.max_layer_size = layer_size;

        size.y = max_layer_height + 2.0f * total_padding;
        // Add room for network info
        if (!disable_info) {
            size.y += 2.0f * to<float>(text_label.getCharacterSize()) + 2.0f * node_radius;
        }

        for (auto& n : nodes) {
            float const node_layer_height = getLayerHeight(architecture.layers[n.layer]);
            float const offset            = (max_layer_height - node_layer_height) * 0.5f;
            n.position.y += offset;
            n.position += sf::Vector2f{background_radius, background_radius};
        }

        // Create connections
        {
            uint32_t connection_idx = 0;
            uint32_t const node_count = nw.info.getNodeCount();
            for (uint32_t i{0}; i < node_count; ++i) {
                const uint32_t connection_count = nw.getNode(i).connection_count;
                for (uint32_t k{0}; k < connection_count; ++k) {
                    auto& c = connections.emplace_back();
                    c.start = nodes[i].position;
                    c.end   = nodes[nw.getConnection(connection_idx).to].position;
                    ++connection_idx;
                }
            }

            assert(connection_idx == nw.connection_count);
        }

        {
            connections_va = sf::VertexArray(sf::Quads, 4 * connections.size());
            uint32_t i{0};
            for (auto const& c : connections) {
                common::Utils::generateLine(connections_va, 4 * i, c.start, c.end, 2.0f, sf::Color::White);
                ++i;
            }
        }

        // Adapt background to new size
        background.setInnerSize(size + 2.0f * Vec2{background_radius, background_radius}, background_outline_thickness);
    }

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();

        target.draw(background, states);

        float const total_padding = background_radius;

        states.transform.translate(total_padding, total_padding);
        target.draw(connections_va, states);

        float const out_radius = node_radius + 3.0f;
        sf::CircleShape shape{out_radius};
        shape.setOrigin(out_radius, out_radius);
        shape.setOutlineColor(sf::Color::White);
        shape.setFillColor(sf::Color::Black);
        shape.setOutlineThickness(2.0f);

        uint32_t node_idx = 0;
        for (auto const& n : nodes) {
            shape.setPosition(n.position);
            target.draw(shape, states);

            float const radius = std::min(n.value.get(), 1.0f) * node_radius;
            sf::CircleShape shape_in{radius};
            shape_in.setOrigin(radius, radius);
            sf::Color const color = (n.value.get() > 0.0f) ? sf::Color{188, 226, 158} : sf::Color{255, 135, 135};
            shape_in.setFillColor(color);
            shape_in.setPosition(n.position);

            target.draw(shape_in, states);

            ++node_idx;
        }

        float label_y = node_radius;
        for (auto const& l : labels) {
            text_label.setString(l);
            auto const bounds = text_label.getGlobalBounds();
            float const width = bounds.width;

            text_label.setPosition(total_padding + label_offset - width - background_radius * 0.5f, total_padding + label_y);

            label_y += 2.0f * node_radius + node_spacing.y;
            target.draw(text_label, states);
        }

        float const max_layer_height = getLayerHeight(architecture.max_layer_size) + 4.0f;

        sf::VertexArray va_line{sf::PrimitiveType::Lines, 2};
        va_line[0].position = {total_padding, total_padding + max_layer_height + 2.0f * node_radius};
        va_line[1].position = {size.x - total_padding, total_padding + max_layer_height + 2.0f * node_radius};
        target.draw(va_line, states);

        text_label.setPosition(total_padding, total_padding + max_layer_height + 2.5f * node_radius);
        text_label.setString("Hidden nodes: " + toString(architecture.hidden_count) +
                             "\nConnections : " + toString(architecture.connection_count));
        if (text_label.getGlobalBounds().width < (size.x - 50.0f)) {
            target.draw(text_label, states);
        }
    }

    void update(nt::Network const& network)
    {
        for (uint32_t i{0}; i < network.connection_count; ++i) {
            auto& c = connections[i];

            c.width = to<float>(network.getConnection(i).value) * 20.0f;
            float const sign = Math::sign(c.width.get());
            float const width = std::max(1.0f, std::min(node_radius, std::abs(c.width.get())));

            sf::Color const color = (sign > 0.0f) ? sf::Color{188, 226, 158} : sf::Color{255, 135, 135};
            common::Utils::generateLine(connections_va, 4 * i, c.start, c.end, width, color);
            ++i;
        }

        {
            network.foreachNode([&](nt::Network::Node const& n, uint32_t i) {
                float value;
                if (i < network.info.inputs) {
                    value = to<float>(n.sum);
                } else {
                    value = to<float>(n.getValue());
                }
                nodes[i].value.setValueInstant(Math::clampAmplitude(value, 1.0f));
            });
        }
    }

    // Utils //////////////////////////

    static uint32_t getMaxDepth(nt::Network const& network)
    {
        return std::max(1u, network.getDepth());
    }

    [[nodiscard]]
    float getLayerHeight(uint32_t height) const
    {
        return to<float>(height) * (2.0f * node_radius + node_spacing.y) - node_spacing.y + node_radius;
    }
};
