using System.Collections.Concurrent;
using SharpNeat.Evaluation;
using SharpNeat.EvolutionAlgorithm;
using SharpNeat.Neat.Genome;

namespace SharpNeat.DistributedServer.Services
{
    public class DistributedGenomeListEvaluator<TGenome, TPhenome> : IGenomeListEvaluator<TGenome>
        where TGenome : class, IGenome
        where TPhenome : IDisposable
    {
        readonly IGenomeDecoder<TGenome, TPhenome> _genomeDecoder;
        readonly IPhenomeEvaluationScheme<TPhenome> _phenomeEvaluationScheme;
        readonly DistributedNeat _distributedNeat;

        public DistributedGenomeListEvaluator(
            IGenomeDecoder<TGenome, TPhenome> genomeDecoder,
            IPhenomeEvaluationScheme<TPhenome> phenomeEvaluationScheme,
            DistributedNeat distributedNeat)
        {
            _genomeDecoder = genomeDecoder;
            _phenomeEvaluationScheme = phenomeEvaluationScheme;
            _distributedNeat = distributedNeat;
        }

        public bool IsDeterministic => _phenomeEvaluationScheme.IsDeterministic;

        public IComparer<FitnessInfo> FitnessComparer => _phenomeEvaluationScheme.FitnessComparer;

        public void Evaluate(IList<TGenome> genomeList)
        {
            var tasks = new ConcurrentBag<Task>();
            Parallel.ForEach(
                SplitIntoSets(genomeList, Program.ChunkSize),
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = 32
                },
                (genomes, _, _) =>
                {
                    var list = new List<NeatGenome<double>>();
                    // Console.WriteLine("Sending " + genomes.Count() + " genomes");
                    foreach (var genome in genomes)
                    {
                        using TPhenome phenome = _genomeDecoder.Decode(genome);
                        if (phenome is null)
                        {
                            genome.FitnessInfo = _phenomeEvaluationScheme.NullFitness;
                        }
                        else
                        {
                            list.Add(genome as NeatGenome<double>);
                        }
                    }

                    tasks.Add(Evaluate(list));
                }
            );
            Task.WaitAll(tasks.ToArray());
        }

        private static IEnumerable<IEnumerable<T>> SplitIntoSets<T>(IEnumerable<T> source, int itemsPerSet)
        {
            var sourceList = source as List<T> ?? source.ToList();
            for (var index = 0; index < sourceList.Count; index += itemsPerSet)
            {
                yield return sourceList.Skip(index).Take(itemsPerSet);
            }
        }

        private Task<List<double>> Evaluate(List<NeatGenome<double>> genomes)
        {
            var task = _distributedNeat.AddJob(genomes, out var id);
            return Task.Run(async () =>
            {
                var result = await TimeoutAfter(task, TimeSpan.FromMinutes(10), id, genomes);
                for (var i = 0; i < genomes.Count; i++)
                {
                    genomes[i].FitnessInfo = new FitnessInfo(result[i]);
                }

                return result;
            });
        }

        private async Task<List<double>> TimeoutAfter(Task<List<double>> task, TimeSpan timeout, string id,
            List<NeatGenome<double>> neatGenomes)
        {
            using var timeoutCancellationTokenSource = new CancellationTokenSource();
            var completedTask = await Task.WhenAny(task, Task.Delay(timeout, timeoutCancellationTokenSource.Token));
            if (completedTask == task)
            {
                timeoutCancellationTokenSource.Cancel();
                return await task; // Very important in order to propagate exceptions
            }

            _distributedNeat.RemoveJob(id);
            return await Evaluate(neatGenomes);
        }

        public bool TestForStopCondition(FitnessInfo fitnessInfo)
        {
            return _phenomeEvaluationScheme.TestForStopCondition(fitnessInfo);
        }
    }
}
