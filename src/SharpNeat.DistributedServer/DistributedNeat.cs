using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Redzen.Random;
using SharpNeat.DistributedServer.Services;
using SharpNeat.Evaluation;
using SharpNeat.EvolutionAlgorithm;
using SharpNeat.EvolutionAlgorithm.Runner;
using SharpNeat.Experiments;
using SharpNeat.IO;
using SharpNeat.IO.Models;
using SharpNeat.Neat;
using SharpNeat.Neat.EvolutionAlgorithm;
using SharpNeat.Neat.Genome;
using SharpNeat.Neat.Genome.Double;
using SharpNeat.Neat.Genome.IO;
using SharpNeat.Neat.Reproduction.Asexual.WeightMutation;
using SharpNeat.NeuralNets.Double;

namespace SharpNeat.DistributedServer
{
    public class DistributedNeat
    {
        private EvolutionAlgorithmRunner _eaRunner;
        private readonly BlockingCollection<TaskGroup> _taskGroups = new();
        private readonly ConcurrentDictionary<string, TaskCompletionSource<List<double>>> _results = new();
        private readonly ILogger<DistributedNeat> _logger;

        public DistributedNeat(ILogger<DistributedNeat> logger)
        {
            _logger = logger;
        }

        public class TaskGroup
        {
            public string Id { get; }
            public List<byte[]> Genomes { get; }

            public TaskGroup(string id, List<byte[]> genomes)
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
                    Program.TaskAssembly
                )
                ?.Unwrap() ?? throw new InvalidOperationException();

            var neatExperiment = factory.CreateExperiment(Program.TaskConfigFile);

            var metaNeatGenome = NeatUtils.CreateMetaNeatGenome(neatExperiment);

            var genomeDecoder =
                NeatGenomeDecoderFactory.CreateGenomeDecoder(
                    neatExperiment.IsAcyclic,
                    neatExperiment.EnableHardwareAcceleratedNeuralNets);

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
                    $"{_eaRunner.EA.Stats.Generation:D},{neatPop.Stats.BestFitness.PrimaryFitness:N3},{neatPop.Stats.MeanFitness:N3},{neatPop.Stats.BestComplexity:N3},{neatPop.Stats.MeanComplexity:N3},{neatPop.Stats.MaxComplexity:N3},{_eaRunner.EA.Stats.TotalEvaluationCount:D},{_eaRunner.EA.Stats.EvaluationsPerSec:N3}";
                
                _logger.LogInformation(line);

                File.AppendAllLines("log.txt", new[] { line });

                NetFile.Save(
                    NeatGenomeConverter.ToNetFileModel(neatPop.BestGenome),
                    _eaRunner.EA.Stats.Generation.ToString());
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

        public void RemoveTaskGroup(string id)
        {
            if (!_results.TryRemove(id, out var taskCompletionSource)) return;
            _logger.LogWarning($"Task group {id} removed");
            taskCompletionSource.SetCanceled();
        }

        public Task<List<double>> AddTaskGroup(List<NeatGenome<double>> genome, out string id)
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
            _taskGroups.Add(new TaskGroup(id, genomeArrays.ToList()));

            return taskCompletionSource.Task;
        }

        public async Task<IEnumerable<TaskGroup>> AcquireTaskGroup(int taskGroupCount)
        {
            _logger.LogInformation($"Acquiring task groups...");
            _logger.LogInformation($"Pending task groups: {_taskGroups.Count}, awaiting task groups: {_results.Count}");
            return await Task.Run(() =>
            {
                // _logger.LogInformation($"Starting to acquire task group");

                var takenTaskGroups = new List<TaskGroup>(taskGroupCount);
                if (_taskGroups.Count > 0)
                {
                    for (var i = 0; i < taskGroupCount; i++)
                    {
                        if (_taskGroups.TryTake(out var taskGroup))
                        {
                            takenTaskGroups.Add(taskGroup);
                        }
                    }
                }

                if (takenTaskGroups.Count == 0)
                {
                    takenTaskGroups.Add(_taskGroups.Take());
                }

                // _logger.LogInformation($"Task groups acquired: {takenTaskGroups.Count}");
                return takenTaskGroups;
            });
        }

        public async Task<TaskGroup> AcquireTaskGroup()
        {
            _logger.LogInformation($"Acquiring task group...");
            _logger.LogInformation($"Pending task groups: {_taskGroups.Count}, awaiting task groups: {_results.Count}");
            return await Task.Run(() => _taskGroups.Take());
        }

        public void SendResult(string id, List<double> fitness)
        {
            _logger.LogInformation($"Result acquired from ${id}");
            _logger.LogInformation($"Pending task groups: {_taskGroups.Count}, awaiting task groups: {_results.Count}");
            if (_results.TryRemove(id, out var taskCompletionSource))
            {
                taskCompletionSource.SetResult(fitness);
            }
        }
    }
}
