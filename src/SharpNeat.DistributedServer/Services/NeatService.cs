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

        public override async Task AcquireTaskGroup(
            IAsyncStreamReader<TaskGroupResult> requestStream,
            IServerStreamWriter<TaskGroupReply> responseStream,
            ServerCallContext context
        )
        {
            var clientName = context.RequestHeaders.Single(e => e.Key == "client-name").Value;
            var taskGroupCount = int.Parse(context.RequestHeaders.Single(e => e.Key == "task-group-count").Value);
            _logger.LogInformation($"Connected to {clientName}, task group count {taskGroupCount}");

            var initialTaskGroup = await _neat.AcquireTaskGroup(taskGroupCount);

            _logger.LogInformation($"Sending task group to {clientName}");
            await SendTaskGroup(responseStream, initialTaskGroup);

            try
            {
                while (await requestStream.MoveNext())
                {
                    // _logger.LogInformation($"Results got: {requestStream.Current.GenomeTaskResults.Count}");
                    foreach (var currentGenomeTaskResult in requestStream.Current.GenomeTaskResults)
                    {
                        _neat.SendResult(currentGenomeTaskResult.Id, currentGenomeTaskResult.Fitness.ToList());
                    }

                    var nextTaskGroup = await _neat.AcquireTaskGroup(taskGroupCount);
                    await SendTaskGroup(responseStream, nextTaskGroup);
                }
            }
            catch
            {
                // ignored
            }

            _logger.LogInformation($"{clientName} disconnected");
        }

        private async Task SendTaskGroup(IServerStreamWriter<TaskGroupReply> responseStream, IEnumerable<DistributedNeat.TaskGroup> taskGroups)
        {
            var genomeTasks = taskGroups
                .Select(taskGroup =>
                    {
                        var genomeTask = new GenomeTask { Id = taskGroup.Id };
                        genomeTask.Genomes.Add(taskGroup.Genomes.Select(ByteString.CopyFrom));
                        return genomeTask;
                    }
                ).ToList();
            var taskGroupReply = new TaskGroupReply
            {
                AssemblyName = "SharpNeat.Tasks",
                TypeName = Program.TaskAssembly,
                ConfigDoc = Program.TaskConfigFile
            };
            taskGroupReply.GenomeTasks.AddRange(genomeTasks);
            await responseStream.WriteAsync(taskGroupReply);
            // _logger.LogInformation($"Tasks sent {taskGroupReply.GenomeTasks.Count}");
        }
    }
}
