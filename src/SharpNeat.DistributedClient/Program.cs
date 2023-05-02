using Grpc.Core;
using Grpc.Net.Client;
using SharpNeat.DistributedProto;
using SharpNeat.Evaluation;
using SharpNeat.Experiments;
using SharpNeat.Neat;
using SharpNeat.Neat.Genome;
using SharpNeat.Neat.Genome.Double;
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
                    Console.WriteLine("Connecting...");
                    // The port number(5000) must match the port of the gRPC server.
                    using var channel = GrpcChannel.ForAddress("http://127.0.0.1:5000");
                    var client = new NeatProtoService.NeatProtoServiceClient(channel);

                    using (var call = client.AcquireJob(new Metadata
                    {
                        new("client-name", Guid.NewGuid().ToString()),
                        // new("batch-size", args[0])
                        // How many tasks to get from server each time
                        new("batch-size", "1")
                    }))
                    {
                        var responseTask = Task.Run(async () =>
                        {
                            // ReSharper disable AccessToDisposedClosure
                            await foreach (var jobReply in call.ResponseStream.ReadAllAsync())
                            {
                                Console.WriteLine("New job acquired");

                                var factory = (INeatExperimentFactory?)Activator.CreateInstance(
                                        jobReply.AssemblyName,
                                        jobReply.TypeName
                                    )
                                    ?.Unwrap() ?? throw new InvalidOperationException();

                                var neatExperiment = factory.CreateExperiment(jobReply.ConfigDoc);
                                var metaNeatGenome = NeatUtils.CreateMetaNeatGenome(neatExperiment);
                                var genomeDecoder = NeatGenomeDecoderFactory.CreateGenomeDecoder(neatExperiment.IsAcyclic);

                                var genomeListEvaluator =
                                    new SerialGenomeListEvaluator<NeatGenome<double>, IBlackBox<double>>(
                                        genomeDecoder,
                                        neatExperiment.EvaluationScheme
                                    );

                                var genomeLists = jobReply.GenomeTasks.Select(
                                    task => task.Genomes.Select(s => NeatGenomeLoader.Load(new MemoryStream(s.ToByteArray()), metaNeatGenome, 0))
                                        .ToList()
                                ).ToList();

                                foreach (var neatGenomes in genomeLists)
                                {
                                    genomeListEvaluator.Evaluate(neatGenomes);
                                }

                                var result = new JobResult
                                {
                                    GenomeTaskResults =
                                    {
                                        genomeLists.Select((genomes, i) =>
                                        {
                                            var genomeTaskResult = new GenomeTaskResult
                                            {
                                                Id = jobReply.GenomeTasks[i].Id,
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

                        // await call.RequestStream.CompleteAsync();
                        await responseTask;
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
