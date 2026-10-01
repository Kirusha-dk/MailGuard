using System.Text.RegularExpressions;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Features;

public sealed class FeatureHasher
{
    private static readonly Regex TokenRegex = new(
        @"[\p{L}\p{Nd}][\p{L}\p{Nd}_\-.@]{1,48}",
        RegexOptions.Compiled);

    private static readonly Regex DomainRegex = new(
        @"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        RegexOptions.Compiled);

    private static readonly Regex UrlDomainRegex = new(
        @"https?://([^/\s:]+)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public int Dimension { get; }

    public FeatureHasher(int dimension = 1 << 17)
    {
        if (dimension <= 1024 || (dimension & (dimension - 1)) != 0)
            throw new ArgumentException("Dimension must be a power of two and > 1024.");

        Dimension = dimension;
    }

    public SparseFeatureVector Extract(
        ParsedEmail email,
        RspamdScanResult? rspamd = null)
    {
        var vector = new SparseFeatureVector();

        AddText(vector, email.Subject, "subject", maxTokens: 120, addBigrams: true);
        AddText(vector, email.Text, "body", maxTokens: 5000, addBigrams: true);

        AddNumeric(vector, "num:url_count", Math.Log(1 + email.UrlCount));
        AddNumeric(vector, "num:attachment_count", Math.Log(1 + email.AttachmentCount));
        AddNumeric(vector, "num:raw_kb", Math.Log(1 + email.RawLength / 1024.0));
        AddBinary(vector, "flag:html", email.HasHtml);

        var fromDomain = ExtractDomain(email.From);
        if (!string.IsNullOrWhiteSpace(fromDomain))
            AddCategorical(vector, "from_domain:" + fromDomain);

        var toDomain = ExtractDomain(email.To);
        if (!string.IsNullOrWhiteSpace(toDomain))
            AddCategorical(vector, "to_domain:" + toDomain);

        foreach (Match match in UrlDomainRegex.Matches(email.Text).Cast<Match>().Take(25))
        {
            var host = match.Groups[1].Value.ToLowerInvariant();
            AddCategorical(vector, "url_domain:" + host);
        }

        if (rspamd is not null)
        {
            AddNumeric(vector, "rspamd:score", Clip(rspamd.Score / 15.0, -3, 3));
            AddNumeric(
                vector,
                "rspamd:score_ratio",
                rspamd.RequiredScore > 0
                    ? Clip(rspamd.Score / rspamd.RequiredScore, -3, 3)
                    : 0);

            AddCategorical(
                vector,
                "rspamd:action:" + NormalizeCategory(rspamd.Action));

            foreach (var symbol in rspamd.Symbols.Take(250))
            {
                var normalized = NormalizeCategory(symbol.Name);
                AddNumeric(
                    vector,
                    "rspamd:symbol:" + normalized,
                    Clip(symbol.Score, -10, 10));

                foreach (var option in symbol.Options.Take(3))
                {
                    var safe = NormalizeOption(option);
                    if (safe.Length > 0)
                        AddCategorical(vector, $"rspamd:option:{normalized}:{safe}");
                }
            }
        }

        return vector;
    }

    private void AddText(
        SparseFeatureVector vector,
        string text,
        string prefix,
        int maxTokens,
        bool addBigrams)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        var tokens = TokenRegex.Matches(text.ToLowerInvariant())
            .Cast<Match>()
            .Select(m => m.Value)
            .Take(maxTokens)
            .ToArray();

        foreach (var token in tokens)
            AddCategorical(vector, $"{prefix}:tok:{token}");

        if (!addBigrams || tokens.Length < 2)
            return;

        var maxBigrams = Math.Min(tokens.Length - 1, 1600);
        for (var i = 0; i < maxBigrams; i++)
            AddCategorical(vector, $"{prefix}:bi:{tokens[i]}_{tokens[i + 1]}");
    }

    private void AddBinary(SparseFeatureVector vector, string name, bool value)
    {
        if (value)
            AddNumeric(vector, name, 1);
    }

    private void AddCategorical(SparseFeatureVector vector, string feature) =>
        AddNumeric(vector, feature, 1);

    private void AddNumeric(
        SparseFeatureVector vector,
        string feature,
        double value)
    {
        var index = Hash(feature) & (Dimension - 1);
        vector.Add(index, value, feature);
    }

    private static int Hash(string value)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (var c in value)
            {
                hash ^= c;
                hash *= 16777619;
            }

            return (int)(hash & 0x7fffffff);
        }
    }

    private static string ExtractDomain(string addressText)
    {
        var match = DomainRegex.Match(addressText);
        return match.Success ? match.Groups[1].Value.ToLowerInvariant() : string.Empty;
    }

    private static string NormalizeCategory(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "none";

        var chars = value.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : '_')
            .ToArray();

        return new string(chars).Trim('_');
    }

    private static string NormalizeOption(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        value = value.ToLowerInvariant();
        value = Regex.Replace(value, @"\d+", "#");
        value = Regex.Replace(
            value,
            @"[^a-zа-яё0-9_.:@#-]+",
            "_",
            RegexOptions.IgnoreCase);

        return value.Length > 80 ? value[..80] : value;
    }

    private static double Clip(double value, double min, double max) =>
        Math.Min(max, Math.Max(min, value));
}
