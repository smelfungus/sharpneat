using Grpc.Core;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.Hosting;

namespace SharpNeat.DistributedServer;

class Program
{
    // Specifies how many genomes should one distributed task contain
    public static int BatchSize { get; private set; }

    public static readonly string TaskAssembly = "SharpNeat.Tasks.Gymnasium.GymnasiumExperimentFactory";
    public static readonly string TaskConfigFile = "config/experiments-config/gymnasium.config.json";
    // public static readonly string TaskAssembly = "SharpNeat.Tasks.BinaryElevenMultiplexer.BinaryElevenMultiplexerExperimentFactory";
    // public static readonly string TaskConfigFile = "config/experiments-config/binary-11-multiplexer.config.json";

    public static void Main(string[] args)
    {
        // ChunkSize = Convert.ToInt32(args[0]);
        BatchSize = 100;
        CreateHostBuilder(args).Build().Run();
    }

    private static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder
                    .ConfigureKestrel(options =>
                    {
                        // Setup a HTTP/2 endpoint without TLS.
                        options.ListenAnyIP(5000, o => o.Protocols = HttpProtocols.Http2);
                        // options.ListenLocalhost(5000, o => o.Protocols = HttpProtocols.Http2);
                    })
                    .UseStartup<Startup>();
            });
}
