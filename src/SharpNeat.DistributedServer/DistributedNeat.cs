using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Redzen.Random;
using SharpNeat.DistributedServer.Services;
using SharpNeat.EvolutionAlgorithm.Runner;
using SharpNeat.Experiments;
using SharpNeat.Neat;
using SharpNeat.Neat.EvolutionAlgorithm;
using SharpNeat.Neat.Genome;
using SharpNeat.Neat.Genome.Double;
using SharpNeat.Neat.Genome.IO;
using SharpNeat.Neat.Reproduction.Asexual.WeightMutation;

namespace SharpNeat.DistributedServer
{
    public class DistributedNeat
    {
        private EvolutionAlgorithmRunner _eaRunner;
        private readonly BlockingCollection<Job> _jobs = new();
        private readonly ConcurrentDictionary<string, TaskCompletionSource<List<double>>> _results = new();
        private readonly ILogger<DistributedNeat> _logger;

        public DistributedNeat(ILogger<DistributedNeat> logger)
        {
            _logger = logger;
        }

        public class Job
        {
            public string Id { get; }
            public List<byte[]> Genomes { get; }

            public Job(string id, List<byte[]> genomes)
            {
                Id = id;
                Genomes = genomes;
            }
        }

        public void StartNeat()
        {
            _logger.LogInformation($"Starting NEAT...");

            var factory = (INeatExperimentFactory?)Activator.CreateInstance(
                    "SharpNeat.Tasks",
                    "SharpNeat.Tasks.BinaryElevenMultiplexer.BinaryElevenMultiplexerExperimentFactory"
                )
                ?.Unwrap() ?? throw new InvalidOperationException();

            var neatExperiment = factory.CreateExperiment("config/experiments-config/binary-11-multiplexer.config.json");

            var metaNeatGenome = NeatUtils.CreateMetaNeatGenome(neatExperiment);

            var neatPop = NeatPopulationFactory<double>.CreatePopulation(
                metaNeatGenome,
                neatExperiment.InitialInterconnectionsProportion,
                neatExperiment.PopulationSize,
                RandomDefaults.CreateRandomSource(42)
            );

            var ea = CreateNeatEvolutionAlgorithm(neatExperiment, neatPop);
            ea.Initialise();

            _eaRunner = new EvolutionAlgorithmRunner(
                ea,
                UpdateScheme.CreateTimeSpanUpdateScheme(TimeSpan.FromMilliseconds(1000))
            );

            // Attach event listeners
            _eaRunner.UpdateEvent += (_, _) =>
            {
                var line =
                    $"bestFitness={neatPop.Stats.BestFitness.PrimaryFitness} stats={_eaRunner.EA.Stats}";
                _logger.LogInformation(line);

                File.AppendAllLines("log.txt", new[] { $"Gen={_eaRunner.EA.Stats.Generation:N0}|BestFitness={neatPop.Stats.BestFitness.PrimaryFitness:N6}|MeanFitness={neatPop.Stats.MeanFitness:N6}|BestComplexity={neatPop.Stats.BestComplexity:N6}|MeanComplexity={neatPop.Stats.MeanComplexity:N6}|MaxComplexity={neatPop.Stats.MaxComplexity:N6}" });
            };

            // Start the algorithm
            _eaRunner.StartOrResume();
        }

        private NeatEvolutionAlgorithm<double> CreateNeatEvolutionAlgorithm(
            INeatExperiment<double> neatExperiment,
            NeatPopulation<double> neatPop)
        {
            // var metaNeatGenome = neatPop.MetaNeatGenome;
            // NeatUtils.ValidateCompatible(neatExperiment, metaNeatGenome);

            var genomeDecoder = NeatGenomeDecoderFactory.CreateGenomeDecoder(
                neatExperiment.IsAcyclic
            );

            var genomeListEvaluator = new DistributedGenomeListEvaluator<NeatGenome<double>, IBlackBox<double>>(
                genomeDecoder,
                neatExperiment.EvaluationScheme,
                this
            );

            var speciationStrategy = NeatUtils.CreateSpeciationStrategy(neatExperiment);

            var weightMutationScheme =
                WeightMutationSchemeFactory.CreateDefaultScheme(neatExperiment.ConnectionWeightScale);

            var ea = new NeatEvolutionAlgorithm<double>(
                neatExperiment.EvolutionAlgorithmSettings,
                genomeListEvaluator,
                speciationStrategy,
                neatPop,
                neatExperiment.ComplexityRegulationStrategy,
                neatExperiment.ReproductionAsexualSettings,
                neatExperiment.ReproductionSexualSettings,
                weightMutationScheme,
                RandomDefaults.CreateRandomSource(42)
            );

            return ea;
        }

        public void RemoveJob(string id)
        {
            if (!_results.TryRemove(id, out var taskCompletionSource)) return;
            _logger.LogWarning($"Task {id} removed");
            taskCompletionSource.SetCanceled();
        }

        public Task<List<double>> AddJob(List<NeatGenome<double>> genome, out string id)
        {
            id = Guid.NewGuid().ToString();
            return Evaluate(id, genome);
        }

        private Task<List<double>> Evaluate(string id, IEnumerable<NeatGenome<double>> genome)
        {
            var genomeArrays = genome.Select(neatGenome =>
            {
                var stream = new MemoryStream();
                NeatGenomeSaver.Save(
                    neatGenome ?? throw new InvalidOperationException(),
                    stream
                );
                var binary = stream.ToArray();
                stream.Flush();
                stream.Close();
                return binary;
            });

            var taskCompletionSource = new TaskCompletionSource<List<double>>();

            _results[id] = taskCompletionSource;
            _jobs.Add(new Job(id, genomeArrays.ToList()));

            return taskCompletionSource.Task;
        }

        public async Task<IEnumerable<Job>> TakeJob(int batchSize)
        {
            return await Task.Run(() =>
            {
                // _logger.LogInformation($"Starting to take jobs");

                var takenJobs = new List<Job>(batchSize);
                if (_jobs.Count > 0)
                {
                    for (var i = 0; i < batchSize; i++)
                    {
                        if (_jobs.TryTake(out var job))
                        {
                            takenJobs.Add(job);
                        }
                    }
                }

                if (takenJobs.Count == 0)
                {
                    takenJobs.Add(_jobs.Take());
                }

                // _logger.LogInformation($"Jobs taken: {takenJobs.Count}");
                return takenJobs;
            });
        }

        public async Task<Job> TakeJob()
        {
            // Console.WriteLine($"{_jobs.Count} new jobs {_results.Count} awaiting jobs");
            return await Task.Run(() => _jobs.Take());
        }

        public void SendResult(string id, List<double> fitness)
        {
            // Console.WriteLine($"{_jobs.Count} new jobs {_results.Count} awaiting jobs");
            if (_results.TryRemove(id, out var taskCompletionSource))
            {
                taskCompletionSource.SetResult(fitness);
            }
        }
    }
}
