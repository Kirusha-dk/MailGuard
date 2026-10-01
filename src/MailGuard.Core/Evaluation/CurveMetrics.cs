namespace MailGuard.Core.Evaluation;

public sealed record LabelScore(bool IsSpam, double Score);
public sealed record RankingMetrics(double RocAuc, double AveragePrecision);
public sealed record ThresholdPoint(
    double Threshold,
    double Recall,
    double FalsePositiveRate,
    double Precision,
    int TruePositives,
    int FalsePositives);

public static class CurveMetrics
{
    public static RankingMetrics Compute(IReadOnlyList<LabelScore> samples)
    {
        if (samples.Count == 0)
            return new RankingMetrics(0, 0);

        return new RankingMetrics(
            ComputeRocAuc(samples),
            ComputeAveragePrecision(samples));
    }

    public static IReadOnlyList<ThresholdPoint> BuildThresholdCurve(
        IReadOnlyList<LabelScore> samples,
        int steps = 100)
    {
        var result = new List<ThresholdPoint>(steps + 1);
        var positives = samples.Count(x => x.IsSpam);
        var negatives = samples.Count - positives;

        for (var i = 0; i <= steps; i++)
        {
            var threshold = i / (double)steps;
            var tp = 0;
            var fp = 0;

            foreach (var sample in samples)
            {
                if (sample.Score < threshold)
                    continue;
                if (sample.IsSpam) tp++;
                else fp++;
            }

            var recall = positives == 0 ? 0 : tp / (double)positives;
            var fpr = negatives == 0 ? 0 : fp / (double)negatives;
            var precision = tp + fp == 0 ? 1 : tp / (double)(tp + fp);
            result.Add(new ThresholdPoint(threshold, recall, fpr, precision, tp, fp));
        }

        return result;
    }

    private static double ComputeAveragePrecision(IReadOnlyList<LabelScore> samples)
    {
        var positives = samples.Count(x => x.IsSpam);
        if (positives == 0)
            return 0;

        var ordered = samples.OrderByDescending(x => x.Score).ToArray();
        var tp = 0;
        var sumPrecision = 0.0;

        for (var i = 0; i < ordered.Length; i++)
        {
            if (!ordered[i].IsSpam)
                continue;

            tp++;
            sumPrecision += tp / (double)(i + 1);
        }

        return sumPrecision / positives;
    }

    private static double ComputeRocAuc(IReadOnlyList<LabelScore> samples)
    {
        var positives = samples.Count(x => x.IsSpam);
        var negatives = samples.Count - positives;
        if (positives == 0 || negatives == 0)
            return 0;

        var ordered = samples.OrderBy(x => x.Score).ToArray();
        var rank = 1;
        var positiveRankSum = 0.0;
        var i = 0;

        while (i < ordered.Length)
        {
            var j = i + 1;
            while (j < ordered.Length && Math.Abs(ordered[j].Score - ordered[i].Score) < 1e-12)
                j++;

            var firstRank = rank;
            var lastRank = rank + (j - i) - 1;
            var averageRank = (firstRank + lastRank) / 2.0;

            for (var k = i; k < j; k++)
            {
                if (ordered[k].IsSpam)
                    positiveRankSum += averageRank;
            }

            rank += j - i;
            i = j;
        }

        return (positiveRankSum - positives * (positives + 1) / 2.0)
               / (positives * (double)negatives);
    }
}
