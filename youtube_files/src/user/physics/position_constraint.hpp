#pragma once
#include "physic_object.hpp"


struct PositionConstraint
{
    siv::ID object_id;
    Vec2    target;
    float   strength;

    PositionConstraint(siv::ID object_id_, Vec2 target_, float strength_)
        : object_id{object_id_}
        , target{target_}
        , strength{strength_}
    {}

    void apply(PhysicObject& object) const
    {
        Vec2 const v = target - object.position;
        Vec2 const c = strength * v;
        object.move(c);
    }
};
