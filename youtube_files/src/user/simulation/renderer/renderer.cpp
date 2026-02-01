#include "renderer.hpp"

#include "user/simulation/agent.hpp"
#include "user/simulation/simulation.hpp"

#include "./vertex_array_utils.hpp"
#include "user/logo/logo.hpp"


Renderer::Renderer()
    : conf{pez::core::getSingleton<SimulationConfiguration>()}
    , background{conf.world_size_f + 2.0f * Vec2{background_margin, background_margin},
                 background_margin,
                 {100, 100, 100}}
    , grid{conf.world_size, 2, 10}
    , population_prey{{graph_width, graph_height}}
    , population_pred{{graph_width, graph_height}}
    , count_food{{graph_width, graph_height}}
    , count_plant{{graph_width, graph_height}}
    , population_update_cooldown(graph_update_delay)
    , agent_va{sf::Quads}
    , light_va{sf::Quads}
    , eyes_va{sf::Quads}
    , food_va{sf::Quads}
    , plant_va{sf::Quads}
    , solver{pez::core::getProcessor<PhysicSolver>()}
    , thread_pool{pez::core::getSingleton<tp::ThreadPool>()}
    , minimap{{graph_width, graph_width}, agent_va, food_va, plant_va}
{
    logo_image = getImageFromLogo(logo.width, logo.height, logo.bytes_per_pixel, logo.pixel_data);
    logo_texture.loadFromImage(logo_image);

    Vec2 const render_size = pez::render::getRenderSize();

    circle_solid_texture  = &pez::resources::getTexture("circle");
    circle_solid_texture->setSmooth(true);
    circle_empty_texture = &pez::resources::getTexture("circle_empty");
    circle_empty_texture->setSmooth(true);
    shadow_texture = &pez::resources::getTexture("shadow");
    shadow_texture->setSmooth(true);
    plant_texture = &pez::resources::getTexture("plant");
    plant_texture->setSmooth(true);
    agent_solid_texture = &pez::resources::getTexture("agent");
    agent_solid_texture->setSmooth(true);
    light_texture = &pez::resources::getTexture("light");
    light_texture->setSmooth(true);

    Vec2 const ui_scale = {conf.ui_scale, conf.ui_scale};

    time_state.setScale(ui_scale);

    minimap.setScale(ui_scale);
    minimap.setPosition({render_size.x - graph_width * ui_scale.x - consts::ui_margin, consts::ui_margin});

    background.setPosition(-Vec2{background_margin, background_margin});
    grid.setColor({110, 110, 110});

    Vec2 const graph_zone_start = render_size - Vec2{
        (graph_width * ui_scale.x) + consts::ui_margin,
        (4.0f * graph_height + 4.0f * consts::ui_margin) * ui_scale.y
    };

    population_prey.setScale(ui_scale);
    population_prey.setColor(conf.prey_color);
    population_prey.background.setFillColor(consts::widget_background_color);
    population_prey.setTitle("Prey population");
    population_prey.setPosition({graph_zone_start.x, graph_zone_start.y});
    population_prey.chart.values.setMaxValuesCount(graph_data_count);
    population_prey.tick_x_period = graph_x_period;

    population_pred.setColor(conf.pred_color);
    population_pred.setScale(ui_scale);
    population_pred.background.setFillColor(consts::widget_background_color);
    population_pred.setTitle("Predator population");
    population_pred.setPosition(Vec2{graph_zone_start.x, graph_zone_start.y + (graph_height + consts::ui_margin) * ui_scale.y});
    population_pred.chart.values.setMaxValuesCount(graph_data_count);
    population_pred.tick_x_period = graph_x_period;

    count_food.setColor(conf.food_color);
    count_food.setScale(ui_scale);
    count_food.background.setFillColor(consts::widget_background_color);
    count_food.setTitle("Food quantity");
    count_food.setPosition(Vec2{graph_zone_start.x, graph_zone_start.y + 2.0f * (graph_height + consts::ui_margin) * ui_scale.y});
    count_food.chart.values.setMaxValuesCount(graph_data_count);
    count_food.tick_x_period = graph_x_period;

    count_plant.setColor(conf.plant_color);
    count_plant.setScale(ui_scale);
    count_plant.background.setFillColor(consts::widget_background_color);
    count_plant.setTitle("Plant count");
    count_plant.setPosition(Vec2{graph_zone_start.x, graph_zone_start.y + 3.0f * (graph_height + consts::ui_margin) * ui_scale.y});
    count_plant.chart.values.setMaxValuesCount(graph_data_count);
    count_plant.tick_x_period = graph_x_period;

    auto const population_lambda = [](float x) {
        return toString(x, 0);
    };
    auto const label_lambda = [](uint32_t x) {
        return toString(x / 10, 0);
    };
    population_prey.current_value_callback = population_lambda;
    population_pred.current_value_callback = population_lambda;
    count_food.current_value_callback = population_lambda;
    count_plant.current_value_callback = population_lambda;
    population_prey.label_callback = label_lambda;
    population_pred.label_callback = label_lambda;
    count_food.label_callback = label_lambda;
    count_plant.label_callback = label_lambda;

    auto_focus.setInterpolationFunction(Interpolation::EaseInOutQuint);
    auto_focus.setSpeed(0.5f);
    auto_focus.setValueInstant(conf.world_size_f * 0.5f);

    auto_zoom.setSpeed(0.25f);
    auto_zoom.setValueInstant(4.2f);
    
    logo_text.setFont(pez::resources::getFont("font"));
    logo_text.setCharacterSize(24);
    logo_text.setFillColor({200, 200, 200});
    logo_text.setScale(ui_scale);
    logo_text.setString("Made by @PezzzasWork");

    // Exit prompt
    //exit_prompt.setX((render_size.x - exit_prompt.size.x) * 0.5f);
    exit_prompt.setScale(ui_scale);
    exit_prompt.setPosition((render_size.x - exit_prompt.size.x * ui_scale.x) * 0.5f, 0.0f);
}

void Renderer::render(pez::render::Context& context)
{
    float constexpr dt = 1.0f / static_cast<float>(consts::tick_rate);

    Vec2 const  render_size = pez::render::getRenderSize();
    auto const& simulation  = pez::core::getProcessor<Simulation>();

    if (observer.over) {
        auto_obs = false;
    }

    if (auto_obs) {
        std::optional<Observer::Event> const event = observer.getEvent(simulation.frame_count);
        if (event) {
            last_event = event;
            // Apply the event
            if (event->selected != siv::InvalidID) {
                // Re access simulation through pez::core because it cannot be const
                pez::core::getProcessor<Simulation>().selected = pez::core::createEntityHandle<Agent>(static_cast<pez::core::ID>(event->selected));
                auto_focus.setSpeed(40.0f);
                auto_zoom = event->zoom;
            } else {
                pez::core::getProcessor<Simulation>().selected = {};
                auto_focus.setSpeed(0.5f);

                zoom_first = event->zoom < auto_zoom.get();
                zoom_done  = false;

                if (zoom_first) {
                    auto_zoom  = event->zoom;
                } else {
                    auto_focus = event->camera_position;
                }
            }
        }

        if (pez::core::isValidHandle(simulation.selected)) {
            auto_focus.setValueInstant(pez::core::get<Agent>(simulation.selected.id.instance_id).position);
        } else {
            float const threshold = 0.5f;
            if (!zoom_done) {
                if (zoom_first) {
                    if (auto_zoom.getElapsedTime() > threshold) {
                        auto_focus = last_event->camera_position;
                        zoom_done = true;
                    }
                } else {
                    if (auto_focus.getElapsedTime() > threshold) {
                        auto_zoom = last_event->zoom;
                        zoom_done = true;
                    }
                }
            }
        }

        pez::render::setFocus(auto_focus);
        pez::render::setZoom(auto_zoom);
    }

    // Update graphs
    if (pez::core::isRunning()) {
        if (population_update_cooldown.updateAutoReset(dt)) {
            population_prey.addValue(static_cast<float>(simulation.prey_count));
            population_pred.addValue(static_cast<float>(simulation.pred_count));
            count_food.addValue(static_cast<float>(pez::core::getCount<Food>()));
            count_plant.addValue(static_cast<float>(pez::core::getCount<Plant>()));
        }
    }

    // Do not render past this point if render is disabled
    if (enable_render) {
        context.draw(background);
        grid.render(context);

        agent_viewer.renderRays(context);

        renderFood(context);
        renderAgents(context);
        renderPlant(context);
    }

    if (render_graphs) {
        context.drawDirect(minimap);

        context.drawDirect(population_prey);
        context.drawDirect(population_pred);
        context.drawDirect(count_food);
        context.drawDirect(count_plant);

        agent_viewer.renderSelected(context);
    }

    time_state.update();
    context.drawDirect(time_state);

    sf::Sprite sprite{logo_texture};
    sprite.setScale(0.25f * conf.ui_scale, 0.25f * conf.ui_scale);
    auto const sprite_bounds = sprite.getGlobalBounds();
    sprite.setPosition(consts::ui_margin, render_size.y - sprite_bounds.height - consts::ui_margin);
    context.drawDirect(sprite);
    logo_text.setOrigin(0.0f, logo_text.getGlobalBounds().height * 0.5f);
    logo_text.setPosition(1.5f * consts::ui_margin + sprite_bounds.width, render_size.y - sprite_bounds.height * 0.5f - consts::ui_margin);
    context.drawDirect(logo_text);

    exit_prompt.update();
    context.drawDirect(exit_prompt);
}

void Renderer::renderAgents(pez::render::Context& context)
{
    uint32_t const agent_count = pez::core::getCount<Agent>();
    agent_va.resize(agent_count * 4);
    light_va.resize(agent_count * 4);
    eyes_va.resize(agent_count * 16);

    auto const& agents = pez::core::getData<Agent>().getData();
    thread_pool.dispatch(agent_count, [&](uint32_t start, uint32_t end) {
        for (uint32_t i{start}; i < end; ++i) {
            Agent const& agent = agents[i];
            addAgentToVertexArray(agent, i);
        }
    });

    {
        sf::RenderStates states;
        states.texture = agent_solid_texture;
        context.draw(agent_va, states);
    }

    {
        sf::RenderStates states;
        states.texture = light_texture;
        states.blendMode = sf::BlendAdd;
        context.draw(light_va, states);
    }

    {
        sf::RenderStates states;
        states.texture = circle_solid_texture;
        context.draw(eyes_va, states);
    }
}

void Renderer::addAgentToVertexArray(const Agent& a, uint32_t i)
{
    const AgentRenderData& render_info  = a.render_data;
    const Vec2             position     = a.position;
    const float            elongation   = std::max(0.75f, render_info.spring_length.getLengthRatio());
    const float            elongation_n = std::max(0.65f, render_info.spring_width.getLengthRatio());
    const Vec2             dir          = 2.1f * render_info.radius * render_info.dir;
    const Vec2             nrm_base     = MathVec2::normal(dir);
    const Vec2             nrm          = nrm_base * elongation_n;
    const Vec2             speed_dir    = dir * elongation;

    constexpr float texture_size{1024.0f};

    const uint32_t agent_index = 4 * i;
    const sf::Color color = {render_info.color.r, render_info.color.g, render_info.color.b};
    agent_va[agent_index + 0].position = position + speed_dir * 0.9f + nrm * 0.9f;
    agent_va[agent_index + 1].position = position + speed_dir * 0.9f - nrm * 0.9f;
    agent_va[agent_index + 2].position = position - speed_dir * 0.9f - nrm * 0.9f;
    agent_va[agent_index + 3].position = position - speed_dir * 0.9f + nrm * 0.9f;
    agent_va[agent_index + 0].color = color;
    agent_va[agent_index + 1].color = color;
    agent_va[agent_index + 2].color = color;
    agent_va[agent_index + 3].color = color;
    agent_va[agent_index + 0].texCoords = {0.0f, 0.0f};
    agent_va[agent_index + 1].texCoords = {texture_size, 0.0f};
    agent_va[agent_index + 2].texCoords = {texture_size, texture_size};
    agent_va[agent_index + 3].texCoords = {0.0f, texture_size};

    uint8_t constexpr light = 150;
    light_va[agent_index + 0].position = position + speed_dir * 0.9f + nrm * 0.9f;
    light_va[agent_index + 1].position = position + speed_dir * 0.9f - nrm * 0.9f;
    light_va[agent_index + 2].position = position - speed_dir * 0.9f - nrm * 0.9f;
    light_va[agent_index + 3].position = position - speed_dir * 0.9f + nrm * 0.9f;
    light_va[agent_index + 0].color =  {light, light, light};
    light_va[agent_index + 1].color =  {light, light, light};
    light_va[agent_index + 2].color =  {light, light, light};
    light_va[agent_index + 3].color =  {light, light, light};
    light_va[agent_index + 0].texCoords = {0.0f, 0.0f};
    light_va[agent_index + 1].texCoords = {texture_size, 0.0f};
    light_va[agent_index + 2].texCoords = {texture_size, texture_size};
    light_va[agent_index + 3].texCoords = {0.0f, texture_size};

    // Generate eyes geometry
    addAgentEyes(a.render_data, render_info.dir, position, i, {elongation, elongation_n});
}

void Renderer::addAgentEyes(const AgentRenderData& render_info, Vec2 dir, Vec2 position, uint32_t index, Vec2 elongation)
{
    const Vec2     nrm       = MathVec2::normal(dir);
    const float    eye_space = 0.21f;
    const float    eye_dist  = render_info.radius * 0.75f * elongation.x;
    const uint32_t i         = 16 * index;

    const Vec2 left_start = position + dir * eye_dist - nrm * eye_space;
    addOneEye(left_start, render_info.look_at, render_info.look_at_dist, i);

    const Vec2 right_start = position + dir * eye_dist + nrm * eye_space;
    addOneEye(right_start, render_info.look_at, render_info.look_at_dist, i + 8);
}

void Renderer::addOneEye(Vec2 position, Vec2 look_at, float dist, uint32_t index)
{
    const float eye_radius = 0.2f;
    const float pupil_coef = std::min(0.7f, std::max(0.25f, dist));

    const Vec2 pupil_pos = position + look_at * eye_radius * (1.0f - pupil_coef);

    constexpr float texture_size{1024.0f};
    VertexArrayUtils::createTexturedQuad(eyes_va, index    , position , eye_radius             , sf::Color::White, texture_size);
    VertexArrayUtils::createTexturedQuad(eyes_va, index + 4, pupil_pos, eye_radius * pupil_coef, sf::Color::Black, texture_size);
}

void Renderer::renderFood(pez::render::Context& context)
{
    auto const& food = pez::core::getData<Food>().getData();
    food_va.resize(food.size() * 4);

    thread_pool.dispatch(to<uint32_t>(food.size()), [&](uint32_t start, uint32_t end) {
        for (uint32_t i{start}; i < end; ++i) {
            Food const& food_element = food[i];
            float const radius = 0.5f * food_element.getRatio(conf);
            uint32_t const food_index{4 * i};
            Vec2 const position = food_element.position;
            food_va[food_index + 0].position = position + Vec2{-radius, -radius};
            food_va[food_index + 1].position = position + Vec2{ radius, -radius};
            food_va[food_index + 2].position = position + Vec2{ radius,  radius};
            food_va[food_index + 3].position = position + Vec2{-radius,  radius};

            food_va[food_index + 0].color = conf.food_color;
            food_va[food_index + 1].color = conf.food_color;
            food_va[food_index + 2].color = conf.food_color;
            food_va[food_index + 3].color = conf.food_color;

            food_va[food_index + 0].texCoords = {0.0f   , 0.0f};
            food_va[food_index + 1].texCoords = {1024.0f, 0.0f};
            food_va[food_index + 2].texCoords = {1024.0f, 1024.0f};
            food_va[food_index + 3].texCoords = {0.0f   , 1024.0f};
        }
    });

    {
        sf::RenderStates states;
        states.texture = shadow_texture;
        context.draw(food_va, states);

        states.texture = circle_empty_texture;
        context.draw(food_va, states);
    }
}

void Renderer::renderPlant(pez::render::Context &context)
{
    plant_va.resize(pez::core::getCount<Plant>() * 8);

    pez::core::parallelForeachEnumerate<Plant>([this](uint32_t i, Plant const& plant) {
        Vec2 const v = plant.getRatio(conf) * plant.direction;
        Vec2 const n = MathVec2::normal(v);
        uint32_t const plant_index{8 * i};
        Vec2 const position = plant.position;
        plant_va[plant_index + 0].position = position + v;
        plant_va[plant_index + 1].position = position + n;
        plant_va[plant_index + 2].position = position - v;
        plant_va[plant_index + 3].position = position - n;
        plant_va[plant_index + 0].color = conf.plant_color;
        plant_va[plant_index + 1].color = conf.plant_color;
        plant_va[plant_index + 2].color = conf.plant_color;
        plant_va[plant_index + 3].color = conf.plant_color;
        plant_va[plant_index + 0].texCoords = {0.0f   , 0.0f};
        plant_va[plant_index + 1].texCoords = {1024.0f, 0.0f};
        plant_va[plant_index + 2].texCoords = {1024.0f, 1024.0f};
        plant_va[plant_index + 3].texCoords = {0.0f   , 1024.0f};

        Vec2 const v_top = plant.getRatio(conf) * plant.direction_top;
        Vec2 const n_top = MathVec2::normal(v_top);
        Vec2 const position_top = plant.position_top;
        plant_va[plant_index + 4].position = position_top + v_top;
        plant_va[plant_index + 5].position = position_top + n_top;
        plant_va[plant_index + 6].position = position_top - v_top;
        plant_va[plant_index + 7].position = position_top - n_top;
        plant_va[plant_index + 4].color = conf.plant_color;
        plant_va[plant_index + 5].color = conf.plant_color;
        plant_va[plant_index + 6].color = conf.plant_color;
        plant_va[plant_index + 7].color = conf.plant_color;
        plant_va[plant_index + 4].texCoords = {0.0f   , 0.0f};
        plant_va[plant_index + 5].texCoords = {1024.0f, 0.0f};
        plant_va[plant_index + 6].texCoords = {1024.0f, 1024.0f};
        plant_va[plant_index + 7].texCoords = {0.0f   , 1024.0f};
    });

    {
        sf::RenderStates states;
        states.texture = plant_texture;
        context.draw(plant_va, states);
    }
}

void Renderer::toggleRender()
{
    enable_render = !enable_render;
}
