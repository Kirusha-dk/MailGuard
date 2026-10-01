namespace MailGuard.Core.Evaluation;

public sealed record ReviewCandidate(
    bool IsSpam,
    double Risk,
    bool RspamdAlreadySpam,
    string Id);

public sealed record ReviewBenchmarkResult(
    int CandidateCount,
    int ReviewCount,
    int TopRiskSpamFound,
    double RandomSpamFoundMean,
    double RandomSpamFoundStdDev,
    double TopRiskPer1000,
    double RandomPer1000Mean,
    double Lift);

public static class ReviewBenchmark
{
    public static ReviewBenchmarkResult Compare(
        IReadOnlyList<ReviewCandidate> all,
        double percent = 1.0,
        int randomRuns = 1000,
        int seed = 1337,
        bool excludeRspamdSpam = true,
        int? budgetBaseCount = null)
    {
        var candidates = all
            .Where(x => !excludeRspamdSpam || !x.RspamdAlreadySpam)
            .ToArray();

        if (candidates.Length == 0)
            return new ReviewBenchmarkResult(0, 0, 0, 0, 0, 0, 0, 0);

        var baseCount = budgetBaseCount ?? candidates.Length;
        if (baseCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(budgetBaseCount));

        var reviewCount = Math.Max(1,
            (int)Math.Ceiling(baseCount * percent / 100.0));
        reviewCount = Math.Min(reviewCount, candidates.Length);

        var topSpam = candidates
            .OrderByDescending(x => x.Risk)
            .ThenBy(x => x.Id, StringComparer.Ordinal)
            .Take(reviewCount)
            .Count(x => x.IsSpam);

        var random = new Random(seed);
        var results = new double[randomRuns];
        var indexes = Enumerable.Range(0, candidates.Length).ToArray();

        for (var run = 0; run < randomRuns; run++)
        {
            for (var i = 0; i < reviewCount; i++)
            {
                var j = random.Next(i, indexes.Length);
                (indexes[i], indexes[j]) = (indexes[j], indexes[i]);
            }

            var found = 0;
            for (var i = 0; i < reviewCount; i++)
            {
                if (candidates[indexes[i]].IsSpam)
                    found++;
            }
            results[run] = found;
        }

        var mean = results.Average();
        var variance = results.Select(x => (x - mean) * (x - mean)).Average();
        var std = Math.Sqrt(variance);
        var topPer1000 = topSpam * 1000.0 / reviewCount;
        var randomPer1000 = mean * 1000.0 / reviewCount;
        var lift = mean <= 1e-12 ? (topSpam > 0 ? double.PositiveInfinity : 0) : topSpam / mean;

        return new ReviewBenchmarkResult(
            candidates.Length,
            reviewCount,
            topSpam,
            mean,
            std,
            topPer1000,
            randomPer1000,
            lift);
    }
}
