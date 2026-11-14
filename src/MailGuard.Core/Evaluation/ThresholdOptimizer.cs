namespace MailGuard.Core.Evaluation;

public sealed record ScoredSample(
    bool IsSpam,
    double Probability,
    bool RspamdAlreadySpam);

public sealed record ThresholdResult(
    double Threshold,
    double Recall,
    double FalsePositiveRate,
    int TruePositives,
    int FalsePositives,
    bool MeetsFalsePositiveConstraint);

public static class ThresholdOptimizer
{
    public static ThresholdResult FindSpamThreshold(
        IReadOnlyList<ScoredSample> samples,
        double maxFalsePositiveRate = 0.005)
    {
        if (samples.Count == 0)
            throw new ArgumentException("Validation set is empty.");
        if (maxFalsePositiveRate < 0 || maxFalsePositiveRate > 1)
            throw new ArgumentOutOfRangeException(nameof(maxFalsePositiveRate));

        if (!double.IsFinite(maxFalsePositiveRate) || samples.Any(x =>
                !double.IsFinite(x.Probability) || x.Probability < 0 || x.Probability > 1))
            throw new ArgumentException("Finite probabilities in [0, 1] are required.");

        var spamTotal = samples.Count(x => x.IsSpam);
        var hamTotal = samples.Count - spamTotal;
        var tp = samples.Count(x => x.IsSpam && x.RspamdAlreadySpam);
        var fp = samples.Count(x => !x.IsSpam && x.RspamdAlreadySpam);
        ThresholdResult At(double threshold) => new(threshold,
            spamTotal == 0 ? 0 : tp / (double)spamTotal,
            hamTotal == 0 ? 0 : fp / (double)hamTotal, tp, fp,
            (hamTotal == 0 ? 0 : fp / (double)hamTotal) <= maxFalsePositiveRate + 1e-12);
        var baseline = At(1.000001);
        ThresholdResult? best = baseline.MeetsFalsePositiveConstraint ? baseline : null;
        foreach (var group in samples.OrderByDescending(x => x.Probability).GroupBy(x => x.Probability))
        {
            foreach (var sample in group)
            {
                if (sample.RspamdAlreadySpam) continue;
                if (sample.IsSpam) tp++; else fp++;
            }
            var result = At(group.Key);
            if (result.MeetsFalsePositiveConstraint && (best is null || result.Recall > best.Recall ||
                (Math.Abs(result.Recall - best.Recall) < 1e-12 && result.FalsePositiveRate < best.FalsePositiveRate) ||
                (Math.Abs(result.Recall - best.Recall) < 1e-12 &&
                 Math.Abs(result.FalsePositiveRate - best.FalsePositiveRate) < 1e-12 && result.Threshold > best.Threshold)))
                best = result;
        }
        return best ?? baseline;
    }
}
