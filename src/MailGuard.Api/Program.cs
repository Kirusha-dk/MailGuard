using MailGuard.Core.Email;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Pipeline;
using MailGuard.Core.Rspamd;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var modelPath = Environment.GetEnvironmentVariable("MAILGUARD_MODEL")
                ?? builder.Configuration["MailGuard:ModelPath"]
                ?? "models/mailguard.mg";

var rspamdUrl = Environment.GetEnvironmentVariable("RSPAMD_URL")
                ?? builder.Configuration["MailGuard:RspamdUrl"]
                ?? "http://localhost:11333/";

MailGuardEngine? engine = null;
HttpClient? rspamdHttp = null;
string? startupError = null;

try
{
    if (File.Exists(modelPath))
    {
        var model = ModelSerializer.Load(modelPath);
        rspamdHttp = new HttpClient
        {
            BaseAddress = new Uri(rspamdUrl.EndsWith('/') ? rspamdUrl : rspamdUrl + "/"),
            Timeout = TimeSpan.FromSeconds(30)
        };

        engine = new MailGuardEngine(
            new EmailParser(),
            new FeatureHasher(model.Classifier.Dimension),
            model,
            new RspamdClient(rspamdHttp));
    }
    else
    {
        startupError = $"Model file not found: {modelPath}";
    }
}
catch (Exception ex)
{
    startupError = ex.Message;
}

app.MapGet("/health", () => Results.Ok(new
{
    status = engine is null ? "degraded" : "ok",
    modelLoaded = engine is not null,
    modelPath,
    rspamdUrl,
    error = startupError
}));

app.MapPost("/classify", async (HttpRequest request, CancellationToken cancellationToken) =>
{
    if (engine is null)
    {
        return Results.Problem(
            title: "MailGuard model is not loaded",
            detail: startupError ?? "Unknown startup error",
            statusCode: StatusCodes.Status503ServiceUnavailable);
    }

    await using var memory = new MemoryStream();
    await request.Body.CopyToAsync(memory, cancellationToken);

    if (memory.Length == 0)
        return Results.BadRequest(new { error = "Request body must contain a raw .eml message." });

    var result = await engine.ClassifyAsync(memory.ToArray(), cancellationToken);
    return Results.Json(result);
});

app.Lifetime.ApplicationStopping.Register(() => rspamdHttp?.Dispose());
app.Run();
