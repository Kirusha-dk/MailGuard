using MailGuard.Core.Domain;

namespace MailGuard.Core.Evaluation;

public sealed class Metrics
{
    public int SpamTotal { get; private set; }
    public int HamTotal { get; private set; }

    public int SpamDetected { get; private set; }
    public int SpamUnsure { get; private set; }
    public int HamFalsePositive { get; private set; }
    public int HamUnsure { get; private set; }

    public int RspamdSpamDetected { get; private set; }
    public int RspamdHamFalsePositive { get; private set; }

    public int Total => SpamTotal + HamTotal;
    public int AutoSpamTotal => SpamDetected + HamFalsePositive;
    public int UnsureTotal => SpamUnsure + HamUnsure;

    public double SpamRecall => Ratio(SpamDetected, SpamTotal);
    public double SpamReviewCoverage => Ratio(SpamDetected + SpamUnsure, SpamTotal);
    public double FalsePositiveRate => Ratio(HamFalsePositive, HamTotal);
    public double SpamPrecision => Ratio(SpamDetected, AutoSpamTotal);
    public double ReviewRate => Ratio(UnsureTotal, Total);

    public double RspamdSpamRecall => Ratio(RspamdSpamDetected, SpamTotal);
    public double RspamdFalsePositiveRate => Ratio(RspamdHamFalsePositive, HamTotal);

    public void Add(bool isSpam, ClassificationResult result)
    {
        if (isSpam)
        {
            SpamTotal++;
            if (result.Decision == MailDecision.Spam)
                SpamDetected++;
            else if (result.Decision == MailDecision.Unsure)
                SpamUnsure++;

            if (result.RspamdAlreadySpam)
                RspamdSpamDetected++;
        }
        else
        {
            HamTotal++;
            if (result.Decision == MailDecision.Spam)
                HamFalsePositive++;
            else if (result.Decision == MailDecision.Unsure)
                HamUnsure++;

            if (result.RspamdAlreadySpam)
                RspamdHamFalsePositive++;
        }
    }

    public string ToReport()
    {
        return $"""
        === MailGuard metrics ===
        Spam total:                {SpamTotal}
        Ham total:                 {HamTotal}

        Rspamd spam recall:        {Pct(RspamdSpamRecall)}
        Rspamd false positive:     {Pct(RspamdFalsePositiveRate)}

        Hybrid spam recall:        {Pct(SpamRecall)}
        Hybrid false positive:     {Pct(FalsePositiveRate)}
        Hybrid spam precision:     {Pct(SpamPrecision)}

        Unsure/review rate:        {Pct(ReviewRate)}
        Spam caught or reviewed:   {Pct(SpamReviewCoverage)}

        Spam detected:             {SpamDetected}/{SpamTotal}
        False positives:           {HamFalsePositive}/{HamTotal}
        """;
    }

    private static double Ratio(int numerator, int denominator) =>
        denominator == 0 ? 0 : numerator / (double)denominator;

    private static string Pct(double value) => $"{value * 100:F2}%";
}
