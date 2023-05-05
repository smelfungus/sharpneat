using System.Diagnostics;
using System.Globalization;
using System.IO.Pipes;
using SharpNeat.Evaluation;

namespace SharpNeat.Tasks.Gymnasium;

public sealed class GymnasiumEpisode
{
    private readonly int _inputCount;
    private readonly int _outputCount;
    private readonly bool _isContinuous;
    private readonly bool _test;

    public GymnasiumEpisode(int inputCount, int outputCount, bool isContinuous, bool test)
    {
        _inputCount = inputCount;
        _outputCount = outputCount;
        _isContinuous = isContinuous;
        _test = test;
    }

    public FitnessInfo Evaluate(IBlackBox<double> phenome)
    {
        // Console.WriteLine("Starting process...");
        var uuid = Guid.NewGuid();

        var start = new ProcessStartInfo
        {
            FileName = @"python3",
            WorkingDirectory = @"./",
            Arguments = string.Format(CultureInfo.InvariantCulture, @"gymnasium/main_unix.py -uuid {0} -render {1} -test False", uuid.ToString(), _test),
            UseShellExecute = false,
            RedirectStandardOutput = false
        };

        var process = Process.Start(start) ?? throw new InvalidOperationException("No process resource is started");
        var totalReward = 0.0;

        try
        {
            // Console.WriteLine($"{uuid}: Opening the pipe...");
            var namedPipeClientStream = new NamedPipeServerStream($"sharpneat.gymnasium.{uuid}.pipe", PipeDirection.InOut);
            namedPipeClientStream.WaitForConnection();
            namedPipeClientStream.ReadMode = PipeTransmissionMode.Byte;
            // Console.WriteLine($"{uuid}: Opened the pipe");

            // Clear any prior agent state.
            phenome.Reset();

            while (true)
            {
                // Determine agent sensor input values.
                // Reset all inputs.
                var inputs = phenome.Inputs.Span;
                inputs.Clear();

                var (observation, rewardArray, doneArray) = ReadObservation(namedPipeClientStream, _inputCount);
                totalReward = rewardArray[0];
                var done = doneArray[0];

                // Console.WriteLine($"{uuid}: Done = {done}");

                if (done == 1)
                {
                    break;
                }

                observation.CopyTo(phenome.Inputs);
                phenome.Activate();

                // var clampedOutputs = outputs.Select(output => Math.Clamp(output, -1.0, 1.0)).ToArray();
                if (_isContinuous)
                {
                    var outputBuffer = new byte[_outputCount * sizeof(float)];
                    var outputs = new double[_outputCount];
                    phenome.Outputs.CopyTo(outputs);
                    Buffer.BlockCopy(Array.ConvertAll(outputs, x => (float)x), 0, outputBuffer, 0, outputBuffer.Length);
                    namedPipeClientStream.Write(outputBuffer, 0, outputBuffer.Length);
                }
                else
                {
                    var maxSigIndex = ReadMaxSigIndex(phenome);
                    var outputBuffer = new byte[sizeof(int)];
                    Buffer.BlockCopy(new[] { maxSigIndex }, 0, outputBuffer, 0, outputBuffer.Length);
                    namedPipeClientStream.Write(outputBuffer, 0, outputBuffer.Length);
                }
            }

            // Console.WriteLine($"{uuid}: Closing the pipe...");
            namedPipeClientStream.Close();
            // Console.WriteLine($"{uuid}: Closed the pipe");
        }
        catch (Exception ex)
        {
            if (!_test)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
        }
        finally
        {
            process.WaitForExit();
        }

        var maskedReward = totalReward < 1 ? Math.Pow(Math.E, totalReward - 1) : totalReward;
        return new FitnessInfo(maskedReward);
    }

    private static (double[] observation, double[] reward, long[] done) ReadObservation(PipeStream namedPipeServerStream, int count)
    {
        var count0 = count * sizeof(double);
        const int count1 = sizeof(double);
        const int count2 = sizeof(long);
        var reader = new BinaryReader(namedPipeServerStream);
        var totalCount = count0 + count1 + count2;
        var inputBuffer = reader.ReadBytes(totalCount);
        if (inputBuffer.Length != totalCount)
        {
            Console.WriteLine($"BinaryReader bytes read: {inputBuffer.Length}/{totalCount}");
        }
        // var inputBuffer = new byte[count0 + count1 + count2];
        // namedPipeServerStream.Read(inputBuffer, 0, inputBuffer.Length);
        var observation = new double[count];
        var reward = new double[1];
        var done = new long[1];
        var offset1 = count0;
        var offset2 = count0 + count1;
        Buffer.BlockCopy(inputBuffer, 0, observation, 0, count0);
        Buffer.BlockCopy(inputBuffer, offset1, reward, 0, count1);
        Buffer.BlockCopy(inputBuffer, offset2, done, 0, count2);

        return (observation, reward, done);
    }

    private static double[] ReadDoubleArray(Stream namedPipeClientStream, int count)
    {
        var inputBuffer = new byte[count * sizeof(double)];
        namedPipeClientStream.Read(inputBuffer, 0, inputBuffer.Length);
        var values = new double[inputBuffer.Length / sizeof(double)];
        Buffer.BlockCopy(inputBuffer, 0, values, 0, values.Length * sizeof(double));
        return values;
    }

    private static float[] ReadFloatArray(Stream namedPipeClientStream, int count)
    {
        var inputBuffer = new byte[count * sizeof(float)];
        namedPipeClientStream.Read(inputBuffer, 0, inputBuffer.Length);
        var values = new float[inputBuffer.Length / sizeof(float)];
        Buffer.BlockCopy(inputBuffer, 0, values, 0, values.Length * sizeof(float));
        return values;
    }

    private static int[] ReadIntArray(Stream namedPipeClientStream, int count)
    {
        var inputBuffer = new byte[count * sizeof(int)];
        namedPipeClientStream.Read(inputBuffer, 0, inputBuffer.Length);
        var values = new int[inputBuffer.Length / sizeof(int)];
        Buffer.BlockCopy(inputBuffer, 0, values, 0, values.Length * sizeof(int));
        return values;
    }

    private int ReadMaxSigIndex(IBlackBox<double> phenome)
    {
        var maxSig = phenome.Outputs.Span[0];
        var maxSigIdx = 0;

        for (var i = 1; i < _outputCount; i++)
        {
            var v = phenome.Outputs.Span[i];
            if (!(v > maxSig)) continue;
            maxSig = v;
            maxSigIdx = i;
        }

        return maxSigIdx;
    }
}
