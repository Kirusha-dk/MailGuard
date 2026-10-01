using System.Text;
using System.Text.Json;
using MailGuard.Core.Ml;

namespace MailGuard.Core.Evaluation;

public sealed record BenchmarkSummary(
    string MailGuardVersion,
    DateTimeOffset CreatedUtc,
    int SpamTotal,
    int HamTotal,
    int RspamdSpamDetected,
    int HybridSpamDetected,
    int RspamdFalsePositives,
    int HybridFalsePositives,
    double RspamdSpamRecall,
    double HybridSpamRecall,
    double RspamdFalsePositiveRate,
    double HybridFalsePositiveRate,
    double HybridPrecision,
    double HybridRocAuc,
    double HybridAveragePrecision,
    double SpamThreshold,
    int ExactLeaksDropped,
    int NearLeaksDropped,
    bool Target90Reached);

public static class BenchmarkReport
{
    public static async Task WriteAsync(
        string jsonPath,
        string markdownPath,
        Metrics metrics,
        RankingMetrics ranking,
        MailGuardModel model,
        int exactLeaksDropped,
        int nearLeaksDropped,
        CancellationToken cancellationToken = default)
    {
        var summary = new BenchmarkSummary(
            model.Version,
            DateTimeOffset.UtcNow,
            metrics.SpamTotal,
            metrics.HamTotal,
            metrics.RspamdSpamDetected,
            metrics.SpamDetected,
            metrics.RspamdHamFalsePositive,
            metrics.HamFalsePositive,
            metrics.RspamdSpamRecall,
            metrics.SpamRecall,
            metrics.RspamdFalsePositiveRate,
            metrics.FalsePositiveRate,
            metrics.SpamPrecision,
            ranking.RocAuc,
            ranking.AveragePrecision,
            model.SpamThreshold,
            exactLeaksDropped,
            nearLeaksDropped,
            metrics.SpamRecall >= 0.90);

        EnsureParent(jsonPath);
        EnsureParent(markdownPath);

        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        await File.WriteAllTextAsync(jsonPath, json + Environment.NewLine, cancellationToken);

        var deltaRecall = metrics.SpamRecall - metrics.RspamdSpamRecall;
        var deltaFp = metrics.HamFalsePositive - metrics.RspamdHamFalsePositive;
        var target = metrics.SpamRecall >= 0.90 ? "REACHED" : "NOT REACHED";

        var md = new StringBuilder();
        md.AppendLine($"# MailGuard {model.Version} benchmark");
        md.AppendLine();
        md.AppendLine($"Generated UTC: {summary.CreatedUtc:O}");
        md.AppendLine();
        md.AppendLine("## Main result");
        md.AppendLine();
        md.AppendLine("| Metric | Rspamd | Rspamd + MailGuard |");
        md.AppendLine("|---|---:|---:|");
        md.AppendLine($"| Spam detected | {metrics.RspamdSpamDetected}/{metrics.SpamTotal} | {metrics.SpamDetected}/{metrics.SpamTotal} |");
        md.AppendLine($"| Spam recall | {metrics.RspamdSpamRecall:P2} | {metrics.SpamRecall:P2} |");
        md.AppendLine($"| False positives | {metrics.RspamdHamFalsePositive}/{metrics.HamTotal} | {metrics.HamFalsePositive}/{metrics.HamTotal} |");
        md.AppendLine($"| False-positive rate | {metrics.RspamdFalsePositiveRate:P3} | {metrics.FalsePositiveRate:P3} |");
        md.AppendLine();
        md.AppendLine($"Recall change vs Rspamd: **{deltaRecall * 100:+0.00;-0.00;0.00} percentage points**.");
        md.AppendLine($"False-positive count change: **{deltaFp:+#;-#;0}**.");
        md.AppendLine($"90% spam-recall target: **{target}**.");
        md.AppendLine();
        md.AppendLine("## Ranking quality");
        md.AppendLine();
        md.AppendLine($"- ROC-AUC: {ranking.RocAuc:F4}");
        md.AppendLine($"- Average precision: {ranking.AveragePrecision:F4}");
        md.AppendLine($"- Hybrid spam precision: {metrics.SpamPrecision:P2}");
        md.AppendLine($"- Selected spam threshold: {model.SpamThreshold:F2}");
        md.AppendLine();
        md.AppendLine("## Leakage control");
        md.AppendLine();
        md.AppendLine($"- Exact train/test duplicates removed from evaluation: {exactLeaksDropped}");
        md.AppendLine($"- Near-duplicates removed from evaluation: {nearLeaksDropped}");
        md.AppendLine();
        md.AppendLine("The target is evaluated only on the remaining held-out messages.");

        await File.WriteAllTextAsync(markdownPath, md.ToString(), cancellationToken);
    }

    private static void EnsureParent(string path)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);
    }
}
