using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using MailGuard.Core.Email;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Pipeline;

// Build this unchanged runner against each Core checkout via -p:MailGuardCorePath=...
// Output contains hashes/counts only. It deliberately has no Rspamd client or training path.
try
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (var i = 0; i < args.Length; i++)
    {
        if (args[i] is not ("--input" or "--output" or "--model" or "--max-messages") || i + 1 >= args.Length)
            throw new ArgumentException("Use --input DIRECTORY --output FILE [--model MODEL.mg] [--max-messages N].");
        if (!options.TryAdd(args[i], args[++i]))
            throw new ArgumentException("Duplicate command option.");
    }
    if (!options.TryGetValue("--input", out var input) || !Directory.Exists(input) ||
        !options.TryGetValue("--output", out var output))
        throw new ArgumentException("Existing --input directory and new --output file are required.");
    var limit = options.TryGetValue("--max-messages", out var count) ? int.Parse(count, CultureInfo.InvariantCulture) : 1000;
    if (limit < 1) throw new ArgumentOutOfRangeException("--max-messages");
    if (File.Exists(output) || File.Exists(output + ".partial"))
        throw new IOException("Output or partial output already exists.");
    var paths = Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories)
        .Where(path => Path.GetExtension(path).Equals(".eml", StringComparison.OrdinalIgnoreCase))
        .OrderBy(path => Path.GetRelativePath(input, path), StringComparer.Ordinal).ToArray();
    if (paths.Length == 0) throw new InvalidDataException("Input contains no .eml messages.");
    var selected = paths.Take(limit).ToArray();
    MailGuardModel? model = null;
    string? modelHash = null;
    if (options.TryGetValue("--model", out var modelPath))
    {
        model = ModelSerializer.Load(modelPath);
        await using var stream = File.OpenRead(modelPath);
        modelHash = Convert.ToHexString(await SHA256.HashDataAsync(stream)).ToLowerInvariant();
    }
    var parser = new EmailParser();
    var features = new FeatureHasher(model?.Classifier.Dimension ?? (1 << 17));
    var engine = model is null ? null : new MailGuardEngine(parser, features, model, null);
    string coreAssemblyHash;
    await using (var coreStream = File.OpenRead(typeof(EmailParser).Assembly.Location))
        coreAssemblyHash = Convert.ToHexString(await SHA256.HashDataAsync(coreStream)).ToLowerInvariant();
    var identities = new HashSet<string>(StringComparer.Ordinal);
    var json = new JsonSerializerOptions(JsonSerializerDefaults.Web);
    await using (var writer = new StreamWriter(new FileStream(output + ".partial", FileMode.CreateNew), new UTF8Encoding(false)))
    {
        await writer.WriteLineAsync(JsonSerializer.Serialize(new
        {
            kind = "metadata", schemaVersion = 1, modelSha256 = modelHash,
            coreAssemblySha256 = coreAssemblyHash,
            commandOptions = new
            {
                inputDirectoryPathSha256 = Hash(Path.GetFullPath(input)), maxMessages = limit,
                modelRequested = model is not null, selectionOrder = "relative-path-ordinal"
            },
            rspamdContext = "absent", classificationAvailable = model is not null,
            featureDimension = features.Dimension, inputMessageCount = paths.Length,
            selectedMessageCount = selected.Length, selectionTruncated = selected.Length < paths.Length,
            maxMessageBytes = 20_000_000, deploymentAcceptanceEstablished = false,
            rawContentSaved = false
        }, json));
        foreach (var path in selected)
        {
            var label = Path.GetFileName(Path.GetDirectoryName(path))?.ToLowerInvariant();
            if (label is not ("spam" or "ham")) label = null;
            string identity;
            byte[]? raw;
            await using (var stream = File.OpenRead(path))
            using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            using (var memory = new MemoryStream())
            {
                var buffer = new byte[81920];
                long length = 0;
                int read;
                while ((read = await stream.ReadAsync(buffer)) != 0)
                {
                    hash.AppendData(buffer, 0, read);
                    length += read;
                    if (length <= 20_000_000) memory.Write(buffer, 0, read);
                }
                identity = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
                raw = length <= 20_000_000 ? memory.ToArray() : null;
            }
            if (!identities.Add(identity))
                throw new InvalidDataException("Duplicate raw-message identity; deduplicate the frozen corpus before comparing.");
            try
            {
                if (raw is null) throw new InvalidDataException("Oversized message.");
                var parsed = parser.Parse(raw);
                var vector = features.Extract(parsed);
                var canonical = new StringBuilder();
                foreach (var item in vector.Values.OrderBy(item => item.Key))
                    canonical.Append(item.Key.ToString(CultureInfo.InvariantCulture)).Append(':')
                        .Append(item.Value.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
                var classification = engine is null ? null : await engine.ClassifyAsync(raw);
                await writer.WriteLineAsync(JsonSerializer.Serialize(new
                {
                    kind = "message", identity, label, parseSuccess = true,
                    parsed = new
                    {
                        textSha256 = Hash(parsed.Text), subjectSha256 = Hash(parsed.Subject),
                        fromSha256 = Hash(parsed.From), toSha256 = Hash(parsed.To),
                        parsed.UrlCount, parsed.AttachmentCount, parsed.HasHtml, parsed.RawLength
                    },
                    featureSha256 = Hash(canonical.ToString()), featureCount = vector.Values.Count,
                    decision = classification?.Decision.ToString(), probability = classification?.SpamProbability
                }, json));
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                await writer.WriteLineAsync(JsonSerializer.Serialize(new
                {
                    kind = "message", identity, label, parseSuccess = false,
                    errorType = ex.GetType().Name
                }, json));
            }
        }
    }
    File.Move(output + ".partial", output, overwrite: false);
    Console.WriteLine(JsonSerializer.Serialize(new { completed = true, messages = selected.Length, rawContentSaved = false }));
    return 0;
}
catch (Exception ex)
{
    // Exception messages may include filesystem paths; do not serialize them into receipts.
    Console.Error.WriteLine($"Runtime regression failed: {ex.GetType().Name}.");
    return 1;
}

static string Hash(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
