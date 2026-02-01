#pragma once
#include "user/configuration.hpp"


/** Dispatches energy between the different needs
 *
 * @note Maybe reserve could be uncapped and affect movement speed.
 *       This may lead to agents having to find a optimal "fit" state where they have energy combined with a low mass.
 */
struct Metabolism
{
    SimulationConfiguration const* conf;

    float reserve; // Maybe a ratio of this?
    float health ;
    float energy ;
    float split   = 0.0f;

    // The baseline consumption
    float reserve_decay;
    float health_decay;
    float split_requirement;

    float dispatch_total  = 0.0f;
    float dispatch_health = 0.0f;
    float dispatch_energy = 0.0f;
    float dispatch_split  = 0.0f;

    explicit
    Metabolism(float health_decay_, float split_requirement_)
        : conf{&pez::core::getSingleton<SimulationConfiguration>()}
        , reserve_decay{conf->agent_reserve_decay}
        , health_decay{health_decay_}
        , split_requirement{split_requirement_}
        , reserve{conf->agent_reserve_max}
        , health{conf->agent_health_max}
        , energy{conf->agent_energy_max}
    {

    }

    /** Adds energy to the reserve. Clamps the reserve to the max
     *
     * @param amount The quantity of energy to add, excess is wasted
     */
    void addToReserve(float amount)
    {
        reserve += amount;
        if (reserve > conf->agent_reserve_max) {
            reserve = conf->agent_reserve_max;
        }
    }

    /** Performs the dispatch
     *
     * @param dt The time step
     */
    void update(float dt)
    {
        // Apply baseline decay
        reserve -= reserve_decay * dt;

        if (reserve < 0.0f) {
            reserve = 0.0f;
            health -= health_decay * reserve_decay * dt;
        }

        dispatch_total = 0.5f;
        dispatch_health = 1.0f - health / conf->agent_health_max;
        dispatch_energy = 1.0f - energy / conf->agent_energy_max;
        // Add a min threshold to avoid asymptote to target value
        dispatch_split  = std::max(0.1f, 1.0f -  split / split_requirement);

        // Dispatch reserve
        float const dispatch_sum = dispatch_health + dispatch_energy + dispatch_split;
        if (dispatch_total > 0.0f && dispatch_sum > 0.1f) {
            // Compute the total reserve flow
            float const requested = conf->agent_metabolism_flow * dispatch_total * dt;
            float const available_quantity = std::min(requested, reserve);
            reserve -= available_quantity;

            // Compute how much of available metabolism will be dispatched to each need
            float const to_health = available_quantity * dispatch_health / dispatch_sum;
            float const to_energy = available_quantity * dispatch_energy / dispatch_sum;
            float const to_split  = available_quantity * dispatch_split / dispatch_sum;

            // Perform the dispatch
            health += to_health;
            energy += to_energy;
            split  += to_split;
        }
    }

    /** Adds the provided quantity to health
     *
     * @param quantity The quantity to add. Clamps health to max value.
     */
    void addHealth(float quantity)
    {
        health += quantity;
        if (health > conf->agent_health_max) {
            health = conf->agent_health_max;
        }
    }

    [[nodiscard]]
    bool splitReady() const
    {
        return split >= split_requirement;
    }

    void performSplit()
    {
        split -= split_requirement;
    }
};