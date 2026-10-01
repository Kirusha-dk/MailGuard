using System.Net.Http.Headers;
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
            throw new InvalidOperationException(
                $"Rspamd returned {(int)response.StatusCode}: {json}");

        return ParseResponse(json);
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
            throw new InvalidOperationException(
                $"Rspamd {endpoint} returned {(int)response.StatusCode}: {body}");
    }

    private static RspamdScanResult ParseResponse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var action = GetString(root, "action");
        var score = GetDouble(root, "score");
        var required = GetDouble(root, "required_score");

        var symbols = new List<RspamdSymbol>();
        if (root.TryGetProperty("symbols", out var symbolsNode)
            && symbolsNode.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in symbolsNode.EnumerateObject())
            {
                var symbolNode = property.Value;
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

        return new RspamdScanResult(action, score, required, symbols);
    }

    private static string GetString(JsonElement node, string name)
    {
        if (!node.TryGetProperty(name, out var value))
            return string.Empty;

        return value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : value.ToString();
    }

    private static double GetDouble(JsonElement node, string name)
    {
        if (!node.TryGetProperty(name, out var value))
            return 0;

        if (value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number))
            return number;

        return double.TryParse(value.ToString(), out number) ? number : 0;
    }
}
