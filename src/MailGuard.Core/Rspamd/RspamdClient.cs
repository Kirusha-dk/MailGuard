using System.Net.Http.Headers;
using System.Globalization;
using System.Text.Json;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Rspamd;

public sealed class RspamdClient : IRspamdScanner
{
    private readonly HttpClient _http;
    private readonly string? _password;

    public RspamdClient(HttpClient http, string? password = null)
    {
        _http = http;
        _password = password;
    }

    public async Task<RspamdScanResult> ScanAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default)
    {
        using var content = new ByteArrayContent(rawMessage);
        content.Headers.ContentType = new MediaTypeHeaderValue("message/rfc822");

        using var request = new HttpRequestMessage(HttpMethod.Post, "checkv2")
        {
            Content = content
        };

        if (!string.IsNullOrWhiteSpace(_password))
            request.Headers.TryAddWithoutValidation("Password", _password);

        using var response = await _http.SendAsync(request, cancellationToken);
        var json = await response.Content.ReadAsStringAsync(cancellationToken);

        if (!response.IsSuccessStatusCode)
            throw new RspamdException(
                $"Rspamd returned HTTP {(int)response.StatusCode}.");

        try
        {
            return ParseResponse(json);
        }
        catch (Exception ex) when (ex is JsonException or InvalidDataException)
        {
            throw new RspamdException("Rspamd returned an invalid scan response.", ex);
        }
    }

    public async Task LearnSpamAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default) =>
        await LearnAsync("learnspam", rawMessage, cancellationToken);

    public async Task LearnHamAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default) =>
        await LearnAsync("learnham", rawMessage, cancellationToken);

    private async Task LearnAsync(
        string endpoint,
        byte[] rawMessage,
        CancellationToken cancellationToken)
    {
        using var content = new ByteArrayContent(rawMessage);
        content.Headers.ContentType = new MediaTypeHeaderValue("message/rfc822");

        using var request = new HttpRequestMessage(HttpMethod.Post, endpoint)
        {
            Content = content
        };

        if (!string.IsNullOrWhiteSpace(_password))
            request.Headers.TryAddWithoutValidation("Password", _password);

        using var response = await _http.SendAsync(request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);

        if (!response.IsSuccessStatusCode)
            throw new RspamdException(
                $"Rspamd {endpoint} returned HTTP {(int)response.StatusCode}.");

        if (body.TrimStart().StartsWith('{'))
        {
            using var document = JsonDocument.Parse(body);
            RejectError(document.RootElement);
        }
    }

    private static RspamdScanResult ParseResponse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        RejectError(root);
        var action = GetString(root, "action");
        if (string.IsNullOrWhiteSpace(action))
            throw new InvalidDataException("Missing Rspamd action.");
        var score = GetDouble(root, "score");
        var required = GetDouble(root, "required_score");

        var symbols = new List<RspamdSymbol>();
        if (root.TryGetProperty("symbols", out var symbolsNode))
        {
            if (symbolsNode.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Invalid Rspamd symbols collection.");
            foreach (var property in symbolsNode.EnumerateObject())
            {
                var symbolNode = property.Value;
                if (symbolNode.ValueKind != JsonValueKind.Object)
                    throw new InvalidDataException("Invalid Rspamd symbol.");
                var symbolScore = GetDouble(symbolNode, "score");
                var options = new List<string>();

                if (symbolNode.TryGetProperty("options", out var optionsNode)
                    && optionsNode.ValueKind == JsonValueKind.Array)
                {
                    foreach (var option in optionsNode.EnumerateArray())
                    {
                        if (option.ValueKind == JsonValueKind.String)
                            options.Add(option.GetString() ?? string.Empty);
                        else
                            options.Add(option.ToString());
                    }
                }

                symbols.Add(new RspamdSymbol(property.Name, symbolScore, options));
            }
        }

        return new RspamdScanResult(action, score, required, symbols)
        {
            IsSkipped = root.TryGetProperty("is_skipped", out var skipped) && skipped.ValueKind == JsonValueKind.True,
            PassthroughModule = root.TryGetProperty("passthrough_module", out var passthrough)
                && passthrough.ValueKind == JsonValueKind.String ? passthrough.GetString() : null
        };
    }

    private static string GetString(JsonElement node, string name)
    {
        if (!node.TryGetProperty(name, out var value))
            return string.Empty;

        return value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : throw new InvalidDataException($"Invalid Rspamd {name}.");
    }

    private static void RejectError(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object ||
            (root.TryGetProperty("error", out var error) && error.ValueKind != JsonValueKind.Null))
            throw new InvalidDataException("Rspamd reported a protocol error.");
    }

    private static double GetDouble(JsonElement node, string name)
    {
        if (node.TryGetProperty(name, out var value))
        {
            if (value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number) && double.IsFinite(number))
                return number;
            if (value.ValueKind == JsonValueKind.String &&
                double.TryParse(value.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out number) &&
                double.IsFinite(number))
                return number;
        }
        throw new InvalidDataException($"Missing or invalid Rspamd {name}.");
    }
}
