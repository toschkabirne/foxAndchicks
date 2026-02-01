#pragma once
#include "engine/common/number_generator.hpp"

#include "genome.hpp"
#include "user/simulation/simulation_configuration.hpp"


namespace nt
{
struct Mutator
{
    /// Mutates a genome using the probabilities defined in conf::mut
    static void mutateGenome(nt::Genome& genome) {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        if (rng.proba(conf.offset_bias_proba)) {
            mutateBiases(genome);
        }

        if (rng.proba(conf.offset_weight_proba)) {
            mutateWeights(genome);
        }

        if (rng.proba(conf.new_node_proba)) {
            newNode(genome);
        }

        if (rng.proba(conf.new_conn_proba)) {
            newConnection(genome);
        }
    }

    static void mutateBiases(nt::Genome& genome)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        Genome::Node& n = pickRandom(genome.nodes);
        if (rng.proba(conf.new_value_proba)) {
            n.bias = rng.getFullRange(conf.weight_range);
        } else {
            n.bias += conf.weight_small_range * rng.getFullRange(conf.weight_range);
        }
    }

    static void mutateWeights(nt::Genome& genome)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        // Nothing to do if no connections
        if (genome.connections.empty()) {
            return;
        }

        Genome::Connection& c = pickRandom(genome.connections);
        if (rng.proba(conf.new_value_proba)) {
            c.weight += rng.getFullRange(conf.weight_range);
        }
    }

    static void newNode(nt::Genome& genome)
    {
        // Nothing to do if no connections
        if (genome.connections.empty()) {
            return;
        }

        uint32_t const connection_idx = getRandIndex(genome.connections.size());
        genome.splitConnection(connection_idx);
    }

    static void newConnection(nt::Genome& genome)
    {
        auto const& conf = pez::core::getSingleton<SimulationConfiguration>();
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        // Pick first random node, input + hidden
        uint32_t const count_1 = genome.info.inputs + genome.info.hidden;
        uint32_t       idx_1   = getRandIndex(count_1);
        // If the picked node is an output, offset it by the number of outputs to land on hidden
        if (idx_1 >= genome.info.inputs && idx_1 < (genome.info.inputs + genome.info.outputs)) {
            idx_1 += genome.info.outputs;
        }
        // Pick second random node, hidden + output
        uint32_t const count_2 = genome.info.hidden + genome.info.outputs;
        // Skip inputs
        uint32_t       idx_2   = getRandIndex(count_2) + genome.info.inputs;

        assert(!genome.isOutput(idx_1));
        assert(!genome.isInput(idx_2));

        // Create the new connection
        if (!genome.tryCreateConnection(idx_1, idx_2, rng.getFullRange(conf.weight_range))) {
            //std::cout << "Cannot create connection " << idx_1 << " -> " << idx_2 << std::endl;
        }
    }

    static uint32_t getRandIndex(uint64_t max_value)
    {
        auto& rng = pez::core::getSingleton<RealNumberGenerator<float>>();

        auto const max_value_f = static_cast<float>(max_value);
        return static_cast<uint32_t>(rng.getUnder(max_value_f));
    }

    template<typename TDataType>
    static TDataType& pickRandom(std::vector<TDataType>& container)
    {
        uint32_t const idx = getRandIndex(container.size());
        return container[idx];
    }
};
}
