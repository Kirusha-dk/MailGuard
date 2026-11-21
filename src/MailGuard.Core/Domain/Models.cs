using System.Text.Json.Serialization;

namespace MailGuard.Core.Domain;

public enum MailDecision
{
    Normal,
    Spam,
    Unsure
}

public sealed record ParsedEmail(
    string Subject,
    string From,
    string To,
    string Text,
    IReadOnlyDictionary<string, string> Headers,
    int UrlCount,
    int AttachmentCount,
    bool HasHtml,
    int RawLength);

public sealed record RspamdSymbol(
    string Name,
    [property: JsonRequired] double Score,
    IReadOnlyList<string> Options);

public sealed record RspamdScanResult(
    string Action,
    [property: JsonRequired] double Score,
    [property: JsonRequired] double RequiredScore,
    IReadOnlyList<RspamdSymbol> Symbols)
{
    public bool IsSkipped { get; init; }
    public string? PassthroughModule { get; init; }
}

public sealed record FeatureExplanation(
    string Feature,
    double Contribution);

public sealed record ClassificationResult(
    MailDecision Decision,
    double SpamProbability,
    double RiskScore,
    bool RspamdAlreadySpam,
    RspamdScanResult? Rspamd,
    IReadOnlyList<FeatureExplanation> Explanations);
