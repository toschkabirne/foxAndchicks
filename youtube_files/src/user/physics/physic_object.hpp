#pragma once
#include "collision_grid.hpp"
#include "engine/common/utils.hpp"
#include "engine/common/math.hpp"
#include "user/simulation/team.hpp"


struct PhysicObject
{
    // Verlet
    Vec2 position          = {0.0f, 0.0f};
    Vec2 last_position     = {0.0f, 0.0f};
    pez::core::ID agent_id = pez::core::EntityID::INVALID_ID;
    Team          team     = Team::None;
    float mass             = 1.0f;

    PhysicObject() = default;

    explicit
    PhysicObject(Vec2 position_, pez::core::ID agent_id_)
        : position{position_}
        , last_position{position_}
        , agent_id{agent_id_}
    {}

    void setPosition(Vec2 pos)
    {
        position      = pos;
        last_position = pos;
    }

    void update(float dt)
    {
        constexpr float friction_coef    = 10.0f;
        const Vec2      last_update_move = position - last_position;
        const Vec2      new_position     = position + last_update_move - last_update_move * (friction_coef * dt);
        last_position                    = position;
        position                         = new_position;
    }

    void stop()
    {
        last_position = position;
    }

    void slowdown(float ratio)
    {
        last_position = last_position + ratio * (position - last_position);
    }

    [[nodiscard]]
    float getSpeed() const
    {
        return MathVec2::length(position - last_position);
    }

    [[nodiscard]]
    Vec2 getVelocity() const
    {
        return position - last_position;
    }

    void addVelocity(Vec2 v)
    {
        last_position -= v;
    }

    void setPositionSameSpeed(Vec2 new_position)
    {
        const Vec2 to_last = last_position - position;
        position           = new_position;
        last_position      = position + to_last;
    }

    void move(Vec2 v)
    {
        position += v;
    }
};
