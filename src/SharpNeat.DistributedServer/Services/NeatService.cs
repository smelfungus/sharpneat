using Google.Protobuf;
using Grpc.Core;
using Microsoft.Extensions.Logging;
using SharpNeat.DistributedProto;

namespace SharpNeat.DistributedServer.Services
{
    public class NeatService : NeatProtoService.NeatProtoServiceBase
    {
        private readonly DistributedNeat _neat;
        private readonly ILogger<NeatService> _logger;

        public NeatService(DistributedNeat neat, ILogger<NeatService> logger)
        {
            _neat = neat;
            _logger = logger;
        }

        public override async Task AcquireJob(
            IAsyncStreamReader<JobResult> requestStream,
            IServerStreamWriter<JobReply> responseStream,
            ServerCallContext context
        )
        {
            var clientName = context.RequestHeaders.Single(e => e.Key == "client-name").Value;
            var batchSize = int.Parse(context.RequestHeaders.Single(e => e.Key == "batch-size").Value);
            _logger.LogInformation($"Connected to {clientName}, batch size {batchSize}");

            var initialJob = await _neat.TakeJob(batchSize);

            _logger.LogInformation($"Sending job to {clientName}");
            await SendJob(responseStream, initialJob);

            try
            {
                while (await requestStream.MoveNext())
                {
                    // _logger.LogInformation($"Results got: {requestStream.Current.GenomeTaskResults.Count}");
                    foreach (var currentGenomeTaskResult in requestStream.Current.GenomeTaskResults)
                    {
                        _neat.SendResult(currentGenomeTaskResult.Id, currentGenomeTaskResult.Fitness.ToList());
                    }

                    var nextJob = await _neat.TakeJob(batchSize);
                    await SendJob(responseStream, nextJob);
                }
            }
            catch
            {
                // ignored
            }

            _logger.LogInformation($"{clientName} disconnected");
        }

        private async Task SendJob(IServerStreamWriter<JobReply> responseStream, IEnumerable<DistributedNeat.Job> jobs)
        {
            var genomeTasks = jobs
                .Select(job =>
                    {
                        var genomeTask = new GenomeTask { Id = job.Id };
                        genomeTask.Genomes.Add(job.Genomes.Select(ByteString.CopyFrom));
                        return genomeTask;
                    }
                ).ToList();
            var jobReply = new JobReply
            {
                AssemblyName = "SharpNeat.Tasks",
                TypeName = "SharpNeat.Tasks.BinaryElevenMultiplexer.BinaryElevenMultiplexerExperimentFactory",
                ConfigDoc = "config/experiments-config/binary-11-multiplexer.config.json"
            };
            jobReply.GenomeTasks.AddRange(genomeTasks);
            await responseStream.WriteAsync(jobReply);
            // _logger.LogInformation($"Jobs sent {jobReply.GenomeTasks.Count}");
        }
    }
}
