using System.Globalization;
using System.Text;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Evaluation;

public static class PredictionCsv
{
    public static async Task WriteAsync(
        string path,
        IReadOnlyList<PredictionRow> rows,
        CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        await using var writer = new StreamWriter(path, false, new UTF8Encoding(true));
        await writer.WriteLineAsync(
            "path,label,fingerprint,decision,local_probability,hybrid_risk," +
            "rspamd_spam,rspamd_action,rspamd_score,top_features");

        foreach (var row in rows)
        {
            var features = string.Join(" | ", row.Result.Explanations.Select(x =>
                $"{x.Feature}:{x.Contribution.ToString("F4", CultureInfo.InvariantCulture)}"));

            await writer.WriteLineAsync(string.Join(",",
                Csv(row.Path),
                row.IsSpam ? "spam" : "ham",
                row.Fingerprint,
                row.Result.Decision,
                row.Result.SpamProbability.ToString("F6", CultureInfo.InvariantCulture),
                row.Result.RiskScore.ToString("F6", CultureInfo.InvariantCulture),
                row.Result.RspamdAlreadySpam ? "1" : "0",
                Csv(row.Result.Rspamd?.Action ?? string.Empty),
                (row.Result.Rspamd?.Score ?? 0).ToString("F4", CultureInfo.InvariantCulture),
                Csv(features)));
        }
    }

    private static string Csv(string value) =>
        "\"" + value.Replace("\"", "\"\"") + "\"";
}

public sealed record PredictionRow(
    string Path,
    bool IsSpam,
    string Fingerprint,
    ClassificationResult Result);
