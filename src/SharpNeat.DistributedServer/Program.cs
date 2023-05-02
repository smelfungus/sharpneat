using Grpc.Core;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.Hosting;

namespace SharpNeat.DistributedServer;

class Program
{
    // Specifies how many genomes should one distributed task contain
    public static int ChunkSize { get; private set; }

    public static void Main(string[] args)
    {
        // ChunkSize = Convert.ToInt32(args[0]);
        ChunkSize = 64;
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
