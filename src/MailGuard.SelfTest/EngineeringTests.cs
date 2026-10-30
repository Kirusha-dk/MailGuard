using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using MailGuard.Core.Domain;
using MailGuard.Core.Email;
using MailGuard.Core.Evaluation;
using MailGuard.Core.Ml;
using MailGuard.Core.Rspamd;

internal static class EngineeringTests
{
    public static IEnumerable<(string Name, Func<Task> Run)> All => new (string, Func<Task>)[]
    {
        ("MIME delimiter lines and attachments", Mime),
        ("legacy charsets and adjacent encoded words", Charsets),
        ("strict Rspamd protocol and invariant numbers", Protocol),
        ("cache corruption, IO failures and cancellation", Cache),
        ("bounded message reader", Reader),
        ("atomic model save and corruption rejection", Serializer),
        ("threshold sweep matches exhaustive oracle", Thresholds)
    };

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static async Task Throws<T>(Func<Task> operation) where T : Exception
    {
        try { await operation(); }
        catch (T) { return; }
        throw new InvalidOperationException($"Expected {typeof(T).Name}.");
    }

    private static Task Mime()
    {
        var raw = Encoding.UTF8.GetBytes("Content-Type: multipart/mixed; boundary=b\r\n\r\n" +
            "preamble\r\n--b\r\nContent-Type: text/plain\r\n\r\n" +
            "keep inline --b marker and line\r\n--bad is body\r\n--b\r\n" +
            "Content-Type: text/plain; name=file.txt\r\n" +
            "Content-Disposition: attachment; filename=file.txt\r\n\r\n" +
            "ATTACHMENT_NOT_BODY\r\n--b--\r\nepilogue");
        var parsed = new EmailParser().Parse(raw);
        Check(parsed.Text.Contains("inline --b marker") && parsed.Text.Contains("--bad is body"), "Boundary substring split body.");
        Check(!parsed.Text.Contains("ATTACHMENT_NOT_BODY") && parsed.AttachmentCount == 1, "Attachment leaked into body.");
        Check(!parsed.Text.Contains("preamble") && !parsed.Text.Contains("epilogue"), "Multipart framing leaked into body.");
        var incomplete = new EmailParser().Parse(Encoding.UTF8.GetBytes(
            "Content-Type: multipart/mixed; boundary=b\n\n--b\nContent-Type: text/plain\n\nlast part"));
        Check(incomplete.Text == "last part", "Unterminated final part was lost.");
        var named = new EmailParser().Parse(Encoding.UTF8.GetBytes(
            "Content-Type: text/plain; name*=utf-8''file.txt\r\n\r\nattachment"));
        Check(named.AttachmentCount == 1 && named.Text.Length == 0, "RFC2231 attachment name was ignored.");
        var deep = string.Concat(Enumerable.Repeat("Content-Type: message/rfc822\r\n\r\n", 22)) + "\r\nbody";
        try { new EmailParser().Parse(Encoding.UTF8.GetBytes(deep)); }
        catch (InvalidDataException) { return Task.CompletedTask; }
        throw new InvalidOperationException("Over-deep MIME was silently truncated.");
    }

    private static Task Charsets()
    {
        var parser = new EmailParser(); // Registers the legacy code-page provider.
        var body = "Привет мир";
        foreach (var charset in new[] { "windows-1251", "koi8-r", "utf-8" })
        {
            var raw = Encoding.ASCII.GetBytes($"Content-Type: text/plain; charset={charset}\r\n\r\n")
                .Concat(Encoding.GetEncoding(charset).GetBytes(body)).ToArray();
            Check(parser.Parse(raw).Text == body, $"Failed to decode {charset}.");
        }
        var words = "Subject: =?UTF-8?B?0J/RgNC4?=\r\n =?UTF-8?B?0LLQtdGC?=\r\n\r\nbody";
        Check(parser.Parse(Encoding.ASCII.GetBytes(words)).Subject == "Привет", "Encoded words retained folding whitespace.");
        var western = Encoding.ASCII.GetBytes("Content-Type: text/plain; charset=windows-1252\r\n\r\n").Concat(new byte[] { 0x80 }).ToArray();
        Check(parser.Parse(western).Text == "€", "Windows-1252 was incorrectly decoded as Latin-1.");
        return Task.CompletedTask;
    }

    private static async Task Protocol()
    {
        var culture = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("ru-RU");
            using var valid = new HttpClient(new ResponseHandler("{\"action\":\"no action\",\"score\":\"1.5\",\"required_score\":6}"))
                { BaseAddress = new Uri("http://localhost/") };
            Check((await new RspamdClient(valid).ScanAsync(Array.Empty<byte>())).Score == 1.5, "Culture changed numeric parsing.");
            foreach (var json in new[] { "{}", "[]", "{\"error\":\"private-message\"}",
                "{\"action\":\"no action\",\"score\":\"NaN\",\"required_score\":6}",
                "{\"action\":\"no action\",\"score\":1}" })
            {
                using var http = new HttpClient(new ResponseHandler(json)) { BaseAddress = new Uri("http://localhost/") };
                await Throws<RspamdException>(async () => await new RspamdClient(http).ScanAsync(Array.Empty<byte>()));
            }
            using var failed = new HttpClient(new ResponseHandler("private-message", HttpStatusCode.BadGateway))
                { BaseAddress = new Uri("http://localhost/") };
            try { await new RspamdClient(failed).ScanAsync(Array.Empty<byte>()); }
            catch (RspamdException ex) { Check(!ex.Message.Contains("private-message"), "Backend payload leaked in error."); }
        }
        finally { CultureInfo.CurrentCulture = culture; }
    }

    private static async Task Cache()
    {
        var directory = Path.Combine(Path.GetTempPath(), "mailguard-cache-" + Guid.NewGuid().ToString("N"));
        try
        {
            var inner = new CounterScanner();
            var cache = new CachedRspamdScanner(inner, directory);
            var raw = Encoding.UTF8.GetBytes("mail");
            await cache.ScanAsync(raw);
            await cache.ScanAsync(raw);
            Check(inner.Calls == 1, "Cache hit rescanned the message.");
            var path = Directory.GetFiles(directory, "*.json", SearchOption.AllDirectories).Single();
            await File.WriteAllTextAsync(path, "{}");
            await cache.ScanAsync(raw);
            Check(inner.Calls == 2, "Semantically invalid cached scan was trusted.");
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            await Throws<OperationCanceledException>(async () => await cache.ScanAsync(raw, cancelled.Token));
            Check(inner.Calls == 2, "Cancelled request reached backend.");
            var other = Encoding.UTF8.GetBytes("second mail");
            var key = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("default\n").Concat(other).ToArray())).ToLowerInvariant();
            await File.WriteAllTextAsync(Path.Combine(directory, key[..2]), "blocks-cache-directory");
            await cache.ScanAsync(other); // Successful scan survives an unwritable cache.
            Check(inner.Calls == 3, "Cache IO test did not scan.");
            Check(!Directory.GetFiles(directory, "*.tmp-*", SearchOption.AllDirectories).Any(), "Temporary cache files leaked.");
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private static async Task Reader()
    {
        using var exact = new MemoryStream(new byte[128]);
        Check((await MessageReader.ReadAsync(exact, 128)).Length == 128, "Exact limit was rejected.");
        using var oversized = new MemoryStream(new byte[129]);
        await Throws<MessageTooLargeException>(async () => await MessageReader.ReadAsync(oversized, 128));
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        using var empty = new MemoryStream();
        await Throws<OperationCanceledException>(async () => await MessageReader.ReadAsync(empty, 128, cancelled.Token));
    }

    private static async Task Serializer()
    {
        var directory = Path.Combine(Path.GetTempPath(), "mailguard-model-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "model.mg");
        try
        {
            var model = new MailGuardModel(new LogisticRegressionModel(2048)) { SpamThreshold = 1.000001 };
            ModelSerializer.Save(path, model);
            var original = File.ReadAllBytes(path);
            Check(ModelSerializer.Load(path).SpamThreshold > 1, "Disabled-SPAM sentinel lost compatibility.");
            model.Classifier.Bias = double.NaN;
            await Throws<InvalidDataException>(() => { ModelSerializer.Save(path, model); return Task.CompletedTask; });
            Check(File.ReadAllBytes(path).SequenceEqual(original), "Invalid save destroyed the previous model.");
            using (var stream = new FileStream(path, FileMode.Append)) stream.WriteByte(42);
            await Throws<InvalidDataException>(() => { ModelSerializer.Load(path); return Task.CompletedTask; });
            File.WriteAllBytes(path, original);
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite))
            {
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
                reader.ReadString(); reader.ReadString();
                using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
                writer.Write(int.MaxValue);
            }
            await Throws<InvalidDataException>(() => { ModelSerializer.Load(path); return Task.CompletedTask; });
        }
        finally { Directory.Delete(directory, true); }
    }

    private static Task Thresholds()
    {
        var random = new Random(51);
        for (var trial = 0; trial < 200; trial++)
        {
            var samples = Enumerable.Range(0, 80).Select(_ => new ScoredSample(random.Next(2) == 1,
                random.Next(11) / 10.0, random.Next(5) == 0)).ToArray();
            var cap = random.Next(11) / 10.0;
            var spam = samples.Count(s => s.IsSpam); var ham = samples.Length - spam;
            var exhaustive = samples.Select(s => s.Probability).Append(1.000001).Distinct().Select(t =>
            {
                var tp = samples.Count(s => s.IsSpam && (s.RspamdAlreadySpam || s.Probability >= t));
                var fp = samples.Count(s => !s.IsSpam && (s.RspamdAlreadySpam || s.Probability >= t));
                return new ThresholdResult(t, spam == 0 ? 0 : tp / (double)spam,
                    ham == 0 ? 0 : fp / (double)ham, tp, fp, (ham == 0 ? 0 : fp / (double)ham) <= cap + 1e-12);
            }).ToArray();
            var expected = exhaustive.Where(r => r.MeetsFalsePositiveConstraint).OrderByDescending(r => r.Recall)
                .ThenBy(r => r.FalsePositiveRate).ThenByDescending(r => r.Threshold).FirstOrDefault()
                ?? exhaustive.Single(r => r.Threshold > 1);
            Check(ThresholdOptimizer.FindSpamThreshold(samples, cap) == expected, "Sweep differs from exhaustive oracle.");
        }
        return Task.CompletedTask;
    }

    private sealed class ResponseHandler(string json, HttpStatusCode status = HttpStatusCode.OK) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token) =>
            Task.FromResult(new HttpResponseMessage(status) { Content = new StringContent(json) });
    }

    private sealed class CounterScanner : IRspamdScanner
    {
        public int Calls;
        public Task<RspamdScanResult> ScanAsync(byte[] rawMessage, CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult(new RspamdScanResult("no action", 0, 6, Array.Empty<RspamdSymbol>()));
        }
    }
}
