using Grpc.Core;
using Grpc.Net.Client;
using SharpNeat.DistributedProto;
using SharpNeat.Experiments;
using SharpNeat.Neat;
using SharpNeat.Neat.Genome.IO;

namespace SharpNeat.DistributedClient
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Initiating client...");

            for (; ; )
            {
                try
                {
                    var address = args.FirstOrDefault("http://127.0.0.1:5000");
                    Console.WriteLine($"Connecting to {address}");
                    // The port number(5000) must match the port of the gRPC server.
                    using var channel = GrpcChannel.ForAddress(address);
                    var client = new NeatProtoService.NeatProtoServiceClient(channel);

                    using (var call = client.AcquireTaskGroup(new Metadata
                    {
                        new("client-name", Guid.NewGuid().ToString()),
                        new("task-group-count", "1")
                    }))
                    {
                        var responseTask = Task.Run(async () =>
                        {
                            // ReSharper disable AccessToDisposedClosure
                            await foreach (var taskGroupReply in call.ResponseStream.ReadAllAsync())
                            {
                                Console.WriteLine($"New batch acquired: {taskGroupReply.GenomeTasks.Sum(task => task.Genomes.Count)} tasks");

                                var factory = (INeatExperimentFactory?)Activator.CreateInstance(
                                        taskGroupReply.AssemblyName,
                                        taskGroupReply.TypeName
                                    )
                                    ?.Unwrap() ?? throw new InvalidOperationException();

                                var neatExperiment = factory.CreateExperiment(taskGroupReply.ConfigDoc);
                                var metaNeatGenome = NeatUtils.CreateMetaNeatGenome(neatExperiment);

                                var genomeListEvaluator = NeatUtils.CreateGenomeListEvaluator(neatExperiment);

                                var genomeLists = taskGroupReply.GenomeTasks.Select(
                                    task => task.Genomes.Select(s => NeatGenomeLoader.Load(new MemoryStream(s.ToByteArray()), metaNeatGenome, 0))
                                        .ToList()
                                ).ToList();

                                Console.WriteLine("Starting evaluation...");
                                foreach (var neatGenomes in genomeLists)
                                {
                                    genomeListEvaluator.Evaluate(neatGenomes);
                                }

                                Console.WriteLine("Evaluation finished");
                                var result = new TaskGroupResult
                                {
                                    GenomeTaskResults =
                                    {
                                        genomeLists.Select((genomes, i) =>
                                        {
                                            var genomeTaskResult = new GenomeTaskResult
                                            {
                                                Id = taskGroupReply.GenomeTasks[i].Id,
                                                Fitness =
                                                {
                                                    genomes.Select(genome => genome.FitnessInfo.PrimaryFitness)
                                                }
                                            };
                                            return genomeTaskResult;
                                        })
                                    }
                                };

                                Console.WriteLine(
                                    $"Sending {result.GenomeTaskResults.Sum(taskResult => taskResult.Fitness.Count)} results");
                                await call.RequestStream.WriteAsync(result);
                            }
                        });

                        await responseTask;
                        await call.RequestStream.CompleteAsync();
                    }

                    Console.WriteLine("Disconnected");
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
        }
    }
}
