using MailGuard.Core.Email;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Pipeline;
using MailGuard.Core.Rspamd;

var builder = WebApplication.CreateBuilder(args);
var limitSetting = Environment.GetEnvironmentVariable("MAILGUARD_MAX_MESSAGE_BYTES");
var maxMessageBytes = MessageReader.DefaultMaxBytes;
if (limitSetting is not null && (!int.TryParse(limitSetting, out maxMessageBytes) || maxMessageBytes <= 0))
    throw new InvalidOperationException("MAILGUARD_MAX_MESSAGE_BYTES must be a positive integer.");
var timeoutSetting = Environment.GetEnvironmentVariable("RSPAMD_TIMEOUT_SECONDS");
var rspamdTimeoutSeconds = 30;
if (timeoutSetting is not null && (!int.TryParse(timeoutSetting, out rspamdTimeoutSeconds) ||
    rspamdTimeoutSeconds is < 1 or > 300))
    throw new InvalidOperationException("RSPAMD_TIMEOUT_SECONDS must be an integer from 1 to 300.");
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = maxMessageBytes);
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
            Timeout = TimeSpan.FromSeconds(rspamdTimeoutSeconds),
            MaxResponseContentBufferSize = 4_000_000
        };

        engine = new MailGuardEngine(
            new EmailParser(),
            new FeatureHasher(model.Classifier.Dimension),
            model,
            new RspamdClient(rspamdHttp, Environment.GetEnvironmentVariable("RSPAMD_PASSWORD")));
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

app.MapGet("/ready", () => engine is null
    ? Results.Json(new { status = "not_ready", modelLoaded = false }, statusCode: 503)
    : Results.Ok(new { status = "ready", modelLoaded = true }));

app.MapPost("/classify", async (HttpRequest request, CancellationToken cancellationToken) =>
{
    if (engine is null)
    {
        return Results.Problem(
            title: "MailGuard model is not loaded",
            detail: startupError ?? "Unknown startup error",
            statusCode: StatusCodes.Status503ServiceUnavailable);
    }

    try
    {
        if (request.ContentLength > maxMessageBytes)
            return Results.StatusCode(StatusCodes.Status413PayloadTooLarge);
        var raw = await MessageReader.ReadAsync(request.Body, maxMessageBytes, cancellationToken);
        if (raw.Length == 0)
            return Results.BadRequest(new { error = "Request body must contain a raw .eml message." });
        var result = await engine.ClassifyAsync(raw, cancellationToken);
        return Results.Json(result);
    }
    catch (MessageTooLargeException)
    {
        return Results.StatusCode(StatusCodes.Status413PayloadTooLarge);
    }
    catch (BadHttpRequestException ex) when (ex.StatusCode == StatusCodes.Status413PayloadTooLarge)
    {
        return Results.StatusCode(StatusCodes.Status413PayloadTooLarge);
    }
    catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
    {
        return Results.Problem(title: "Rspamd request timed out", statusCode: 504);
    }
    catch (Exception ex) when (ex is InvalidDataException or System.Text.RegularExpressions.RegexMatchTimeoutException)
    {
        return Results.Problem(title: "Message cannot be parsed within the supported limits", statusCode: 400);
    }
    catch (Exception ex) when (ex is RspamdException or HttpRequestException)
    {
        app.Logger.LogWarning("Rspamd request failed ({ExceptionType})", ex.GetType().Name);
        return Results.Problem(title: "Rspamd is unavailable or returned an invalid response", statusCode: 502);
    }
});

app.Lifetime.ApplicationStopping.Register(() => rspamdHttp?.Dispose());
app.Run();
