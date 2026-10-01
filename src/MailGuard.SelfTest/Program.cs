using System.Text;
using MailGuard.Core.Domain;
using MailGuard.Core.Email;
using MailGuard.Core.Evaluation;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Pipeline;
using MailGuard.Core.Rspamd;

var tests = new List<(string Name, Func<Task> Run)>
{
    ("email parser", TestEmailParser),
    ("feature extraction", TestFeatures),
    ("threshold optimizer", TestThresholds),
    ("model serializer", TestSerializer),
    ("rspamd protection", TestRspamdProtection)
};

var passed = 0;
foreach (var test in tests)
{
    try
    {
        await test.Run();
        passed++;
        Console.WriteLine($"PASS {test.Name}");
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"FAIL {test.Name}: {ex.Message}");
        return 1;
    }
}

Console.WriteLine($"MailGuard self-test: {passed}/{tests.Count} passed.");
return 0;

static Task TestEmailParser()
{
    var raw = Encoding.UTF8.GetBytes(
        "Subject: =?UTF-8?B?0KLQtdGB0YI=?=\r\n" +
        "From: sender@example.com\r\n" +
        "To: user@example.net\r\n" +
        "MIME-Version: 1.0\r\n" +
        "Content-Type: multipart/mixed; boundary=\"b\"\r\n\r\n" +
        "--b\r\n" +
        "Content-Type: text/html; charset=utf-8\r\n" +
        "Content-Transfer-Encoding: quoted-printable\r\n\r\n" +
        "<html><body>Hello =D0=BC=D0=B8=D1=80 https://example.com/a</body></html>\r\n" +
        "--b\r\n" +
        "Content-Type: application/octet-stream; name=\"x.bin\"\r\n" +
        "Content-Disposition: attachment; filename=\"x.bin\"\r\n" +
        "Content-Transfer-Encoding: base64\r\n\r\n" +
        "AQID\r\n" +
        "--b--\r\n");

    var parsed = new EmailParser().Parse(raw);
    Assert(parsed.Subject == "Тест", "RFC 2047 subject was not decoded.");
    Assert(parsed.From.Contains("example.com"), "From header missing.");
    Assert(parsed.HasHtml, "HTML flag missing.");
    Assert(parsed.AttachmentCount == 1, "Attachment count is wrong.");
    Assert(parsed.UrlCount == 1, "URL count is wrong.");
    Assert(parsed.Text.Contains("Hello"), "Body text missing.");
    return Task.CompletedTask;
}

static Task TestFeatures()
{
    var email = new ParsedEmail(
        "Free prize",
        "promo@example.com",
        "user@example.net",
        "Click https://promo.example/win now",
        new Dictionary<string, string>(),
        1,
        0,
        false,
        120);

    var rspamd = new RspamdScanResult(
        "no action",
        3.2,
        15,
        new[] { new RspamdSymbol("TEST_SYMBOL", 1.5, Array.Empty<string>()) });

    var vector = new FeatureHasher().Extract(email, rspamd);
    Assert(vector.Values.Count > 5, "Too few features were extracted.");
    return Task.CompletedTask;
}

static Task TestThresholds()
{
    var samples = new[]
    {
        new ScoredSample(true, .95, false),
        new ScoredSample(true, .80, false),
        new ScoredSample(false, .20, false),
        new ScoredSample(false, .10, false)
    };

    var result = ThresholdOptimizer.FindSpamThreshold(samples, 0);
    Assert(result.TruePositives == 2, "Threshold should keep both spam samples.");
    Assert(result.FalsePositives == 0, "Threshold created a false positive.");
    return Task.CompletedTask;
}

static Task TestSerializer()
{
    var classifier = new LogisticRegressionModel(2048);
    classifier.Weights[12] = 1.25;
    classifier.Bias = -0.4;

    var model = new MailGuardModel(classifier)
    {
        Version = "selftest",
        SpamThreshold = .91,
        HamThreshold = .12
    };
    model.TrainingExactHashes.Add("abc");

    var path = Path.Combine(Path.GetTempPath(), "mailguard-" + Guid.NewGuid().ToString("N") + ".mg");
    try
    {
        ModelSerializer.Save(path, model);
        var loaded = ModelSerializer.Load(path);
        Assert(loaded.Version == "selftest", "Model version changed.");
        Assert(Math.Abs(loaded.SpamThreshold - .91) < 1e-9, "Spam threshold changed.");
        Assert(Math.Abs(loaded.Classifier.Weights[12] - 1.25) < 1e-6, "Weight changed.");
        Assert(loaded.TrainingExactHashes.Count == 1, "Training hash missing.");
    }
    finally
    {
        if (File.Exists(path))
            File.Delete(path);
    }

    return Task.CompletedTask;
}

static async Task TestRspamdProtection()
{
    var classifier = new LogisticRegressionModel(2048)
    {
        Bias = -20
    };
    var model = new MailGuardModel(classifier)
    {
        SpamThreshold = .99,
        ProtectRspamdSpam = true
    };

    var engine = new MailGuardEngine(
        new EmailParser(),
        new FeatureHasher(2048),
        model,
        new FakeScanner());

    var raw = Encoding.UTF8.GetBytes(
        "Subject: normal\r\nFrom: a@example.com\r\nTo: b@example.net\r\n\r\nhello");

    var result = await engine.ClassifyAsync(raw);
    Assert(result.Decision == MailDecision.Spam, "Protected Rspamd spam was downgraded.");
    Assert(result.RspamdAlreadySpam, "Rspamd spam flag missing.");
}

static void Assert(bool condition, string message)
{
    if (!condition)
        throw new InvalidOperationException(message);
}

sealed class FakeScanner : IRspamdScanner
{
    public Task<RspamdScanResult> ScanAsync(byte[] rawMessage, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new RspamdScanResult(
            "reject",
            20,
            15,
            Array.Empty<RspamdSymbol>()));
    }
}
