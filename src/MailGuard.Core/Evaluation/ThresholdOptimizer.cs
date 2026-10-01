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

        var thresholds = samples
            .Select(x => Math.Clamp(x.Probability, 0, 1))
            .Distinct()
            .Append(1.000001)
            .OrderByDescending(x => x)
            .ToArray();

        ThresholdResult? bestFeasible = null;
        ThresholdResult? baseline = null;

        foreach (var threshold in thresholds)
        {
            var result = EvaluateAt(samples, threshold, maxFalsePositiveRate);
            if (threshold > 1)
                baseline = result;

            if (!result.MeetsFalsePositiveConstraint)
                continue;

            if (bestFeasible is null
                || result.Recall > bestFeasible.Recall
                || (Math.Abs(result.Recall - bestFeasible.Recall) < 1e-12
                    && result.FalsePositiveRate < bestFeasible.FalsePositiveRate)
                || (Math.Abs(result.Recall - bestFeasible.Recall) < 1e-12
                    && Math.Abs(result.FalsePositiveRate - bestFeasible.FalsePositiveRate) < 1e-12
                    && result.Threshold > bestFeasible.Threshold))
            {
                bestFeasible = result;
            }
        }

        return bestFeasible
               ?? baseline
               ?? EvaluateAt(samples, 1.000001, maxFalsePositiveRate);
    }

    private static ThresholdResult EvaluateAt(
        IReadOnlyList<ScoredSample> samples,
        double threshold,
        double maxFalsePositiveRate)
    {
        var spamTotal = samples.Count(x => x.IsSpam);
        var hamTotal = samples.Count - spamTotal;
        var tp = 0;
        var fp = 0;

        foreach (var sample in samples)
        {
            var predicted = sample.RspamdAlreadySpam
                            || sample.Probability >= threshold;
            if (predicted && sample.IsSpam)
                tp++;
            else if (predicted && !sample.IsSpam)
                fp++;
        }

        var recall = spamTotal == 0 ? 0 : tp / (double)spamTotal;
        var fpr = hamTotal == 0 ? 0 : fp / (double)hamTotal;

        return new ThresholdResult(
            threshold,
            recall,
            fpr,
            tp,
            fp,
            fpr <= maxFalsePositiveRate + 1e-12);
    }
}
