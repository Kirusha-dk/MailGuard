using System.Text.Json;
using MailGuard.Core.Email;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Pipeline;
using MailGuard.Core.Rspamd;

namespace MailGuard.Cli;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true
    };

    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return 0;
        }

        try
        {
            return args[0].ToLowerInvariant() switch
            {
                "classify" => await ClassifyAsync(args[1..]),
                "parse" => Parse(args[1..]),
                "model-info" => ModelInfo(args[1..]),
                _ => Unknown(args[0])
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("MailGuard error: " + ex.Message);
            return 1;
        }
    }

    private static async Task<int> ClassifyAsync(string[] args)
    {
        var options = ParseOptions(args);
        var input = Require(options, "input");
        var modelPath = Require(options, "model");
        var raw = await File.ReadAllBytesAsync(input);

        var model = ModelSerializer.Load(modelPath);
        var rspamdUrl = Get(options, "rspamd")
                        ?? Environment.GetEnvironmentVariable("RSPAMD_URL");

        IRspamdScanner? scanner = null;
        HttpClient? http = null;

        if (!string.IsNullOrWhiteSpace(rspamdUrl))
        {
            http = new HttpClient
            {
                BaseAddress = new Uri(EnsureSlash(rspamdUrl)),
                Timeout = TimeSpan.FromSeconds(30)
            };
            scanner = new RspamdClient(http);
        }

        try
        {
            var engine = new MailGuardEngine(
                new EmailParser(),
                new FeatureHasher(model.Classifier.Dimension),
                model,
                scanner);

            var result = await engine.ClassifyAsync(raw);
            Console.WriteLine(JsonSerializer.Serialize(result, Json));
            return 0;
        }
        finally
        {
            http?.Dispose();
        }
    }

    private static int Parse(string[] args)
    {
        var options = ParseOptions(args);
        var input = Require(options, "input");
        var raw = File.ReadAllBytes(input);
        var parsed = new EmailParser().Parse(raw);
        Console.WriteLine(JsonSerializer.Serialize(parsed, Json));
        return 0;
    }

    private static int ModelInfo(string[] args)
    {
        var options = ParseOptions(args);
        var modelPath = Require(options, "model");
        var model = ModelSerializer.Load(modelPath);

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            model.Version,
            Dimension = model.Classifier.Dimension,
            model.SpamThreshold,
            model.HamThreshold,
            model.ProtectRspamdSpam,
            TrainingExactHashes = model.TrainingExactHashes.Count,
            TrainingSimHashes = model.TrainingSimHashes.Count
        }, Json));

        return 0;
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (!arg.StartsWith("--", StringComparison.Ordinal))
                continue;

            var name = arg[2..];
            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                result[name] = "true";
            else
                result[name] = args[++i];
        }

        return result;
    }

    private static string Require(IReadOnlyDictionary<string, string> options, string name) =>
        Get(options, name) ?? throw new ArgumentException($"Missing --{name}.");

    private static string? Get(IReadOnlyDictionary<string, string> options, string name) =>
        options.TryGetValue(name, out var value) ? value : null;

    private static string EnsureSlash(string value) =>
        value.EndsWith('/') ? value : value + "/";

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"Unknown command: {command}");
        PrintHelp();
        return 2;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
        MailGuard CLI

          parse --input message.eml

          classify --input message.eml --model models/mailguard.mg
                   [--rspamd http://127.0.0.1:11333/]

          model-info --model models/mailguard.mg
        """);
    }
}
