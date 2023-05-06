using SharpNeat.Evaluation;
using SharpNeat.Experiments;
using SharpNeat.IO;
using SharpNeat.IO.Models;
using SharpNeat.Neat;
using SharpNeat.Neat.Genome.Double;
using SharpNeat.Neat.Genome.IO;
using SharpNeat.Tasks.Gymnasium;

namespace SharpNeat.Sandbox
{
    class Program
    {
        static void Main(string[] args)
        {
            var factory = new GymnasiumExperimentFactory();

            var neatExperiment = factory.CreateExperiment("config/experiments-config/gymnasium.config.json");
            var metaNeatGenome = NeatUtils.CreateMetaNeatGenome(neatExperiment);

            var genomeDecoder =
                NeatGenomeDecoderFactory.CreateGenomeDecoder(
                    neatExperiment.IsAcyclic,
                    neatExperiment.EnableHardwareAcceleratedNeuralNets);

            // Create a genomeList evaluator, and return.
            var genomeListEvaluator = GenomeListEvaluatorFactory.CreateEvaluator(
                genomeDecoder,
                neatExperiment.EvaluationScheme,
                1);

            var genome = NeatGenomeConverter.ToNeatGenome(NetFile.Load("17979"), metaNeatGenome, 1);

            var episode = new GymnasiumEpisode(24, 4, true, true);
            episode.Evaluate(genomeDecoder.Decode(genome));
        }
    }
}
