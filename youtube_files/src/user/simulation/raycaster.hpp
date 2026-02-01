#pragma once
#include "engine/common/math.hpp"
#include "user/physics/collision_grid.hpp"


struct Raycaster
{
    struct Ray
    {
        Vec2  direction = {};
        float length    = 0.0f;
        Team  team      = Team::None;
    };

    //float fov = Math::PI * 0.5f;

    PhysicSolver const&            solver;
    CollisionGrid const&           grid;
    SimulationConfiguration const& conf;

    Raycaster()
        : solver{pez::core::getProcessor<PhysicSolver>()}
        , grid{solver.grid}
        , conf{pez::core::getSingleton<SimulationConfiguration>()}
    {
    }

    static Ray getHitDist(PhysicObject const& obj, Vec2 d, Vec2 p)
    {
        const Vec2 u0 = obj.position - p;
        const Vec2 u1 = MathVec2::dot(u0, d) * d;
        const Vec2 u2 = u0 - u1;
        const float d2 = MathVec2::length2(u2);
        if (d2 <= 0.25f) {
            const float m = std::sqrt(0.25f - d2);
            return {d, MathVec2::length(u1) - m, obj.team};
        }
        return {{}, -1.0f, Team::None};
    }

    [[nodiscard]]
    Ray castRay(Vec2 position, Vec2 direction, pez::core::ID agent_id) const
    {
        constexpr float eps{0.0001f};

        Ray result{direction, 0.0f};
        IVec2 cell_p{to<IVec2>(position)};
        const IVec2 step(direction.x < 0.0f ? -1 : 1, direction.y < 0.0f ? -1 : 1);

        const Vec2  inv_d(1.0f / (direction.x == 0.0f ? eps : direction.x),
                          1.0f / (direction.y == 0.0f ? eps : direction.y));

        const float t_dx = std::abs(inv_d.x);
        const float t_dy = std::abs(inv_d.y);
        float t_max_x = ((cell_p.x + (step.x > 0)) - position.x) * inv_d.x;
        float t_max_y = ((cell_p.y + (step.y > 0)) - position.y) * inv_d.y;
        const int32_t max_width  = grid.width - 1;
        const int32_t max_height = grid.height - 1;

        float const max_dist = conf.agent_ray_max_dist;

        while (result.length < max_dist) {
            const uint32_t b = t_max_x < t_max_y;
            // Advance in grid
            result.length = b * t_max_x + (!b) * t_max_y;
            t_max_x += t_dx * b;
            t_max_y += t_dy * (!b);
            cell_p.x += step.x * b;
            cell_p.y += step.y * (!b);

            if (cell_p.x >= 0 && cell_p.x < max_width && cell_p.y >= 0 && cell_p.y < max_height) {
                const CollisionCell& cell = grid.get(cell_p);
                float min_dist = -1.0f;
                for (uint32_t i{0}; i < cell.objects_count; ++i) {
                    auto const& physicsObject = solver.getObjectRaw(cell.objects[i]);
                    // Avoid self intersection
                    if (physicsObject.agent_id != agent_id) {
                        Ray const hit_result = getHitDist(solver.getObjectRaw(cell.objects[i]), direction, position);
                        if (hit_result.length != -1.0f) {
                            if (min_dist == -1.0f || hit_result.length < min_dist) {
                                result = hit_result;
                                min_dist = hit_result.length;
                            }
                        }
                    }
                }
                if (min_dist != -1.0f) {
                    break;
                }
            } else {
                break;
            }
        }

        ///@TODO Just to be sure, maybe remove it later
        result.length = std::min(result.length, max_dist);

        return result;
    }

    void cast(Vec2 position, float angle, float fov, std::vector<Ray>& rays, pez::core::ID agent_id) const
    {
        uint32_t const ray_count = conf.ray_count;

        float const half_fov{fov * 0.5f};
        float const start_angle{angle - half_fov};
        float const da{fov / to<float>(ray_count - 1)};

        for (uint32_t i{0}; i < ray_count; ++i) {
            float const a{start_angle + to<float>(i) * da};
            Vec2 const d{std::cos(a), std::sin(a)};
            rays[i] = castRay(position, d, agent_id);
        }
    }
};