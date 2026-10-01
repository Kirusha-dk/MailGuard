namespace MailGuard.Core.Ml;

public sealed class MailGuardModel
{
    public string Version { get; set; } = "0.4.5";
    public LogisticRegressionModel Classifier { get; }
    public double SpamThreshold { get; set; } = 0.80;
    public double HamThreshold { get; set; } = 0.20;
    public bool ProtectRspamdSpam { get; set; } = true;
    public List<string> TrainingExactHashes { get; } = new();
    public List<ulong> TrainingSimHashes { get; } = new();

    public MailGuardModel(LogisticRegressionModel classifier)
    {
        Classifier = classifier;
    }
}
