using MailGuard.Core.Domain;
using MailGuard.Core.Email;
using MailGuard.Core.Features;
using MailGuard.Core.Ml;
using MailGuard.Core.Rspamd;

namespace MailGuard.Core.Pipeline;

public sealed class MailGuardEngine
{
    private readonly EmailParser _parser;
    private readonly FeatureHasher _features;
    private readonly MailGuardModel _model;
    private readonly IRspamdScanner? _rspamd;

    public MailGuardEngine(
        EmailParser parser,
        FeatureHasher features,
        MailGuardModel model,
        IRspamdScanner? rspamd)
    {
        _parser = parser;
        _features = features;
        _model = model;
        _rspamd = rspamd;
    }

    public async Task<ClassificationResult> ClassifyAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default)
    {
        var email = _parser.Parse(rawMessage);
        var rspamd = _rspamd is null
            ? null
            : await _rspamd.ScanAsync(rawMessage, cancellationToken);

        var vector = _features.Extract(email, rspamd);
        var probability = _model.Classifier.PredictProbability(vector);
        var rspamdAlreadySpam = IsRspamdSpam(rspamd);

        MailDecision decision;
        if (_model.ProtectRspamdSpam && rspamdAlreadySpam)
            decision = MailDecision.Spam;
        else if (probability >= _model.SpamThreshold)
            decision = MailDecision.Spam;
        else if (probability <= _model.HamThreshold)
            decision = MailDecision.Normal;
        else
            decision = MailDecision.Unsure;

        var risk = rspamdAlreadySpam ? 1.0 : probability;
        var explanations = Explain(vector, 8);

        return new ClassificationResult(
            decision,
            probability,
            risk,
            rspamdAlreadySpam,
            rspamd,
            explanations);
    }

    public static bool IsRspamdSpam(RspamdScanResult? result)
    {
        if (result is null)
            return false;

        var action = result.Action.Trim().ToLowerInvariant();
        return action is "reject"
            or "add header"
            or "rewrite subject"
            or "quarantine"
            or "discard";
    }

    private IReadOnlyList<FeatureExplanation> Explain(
        SparseFeatureVector vector,
        int take)
    {
        var list = new List<FeatureExplanation>();
        foreach (var pair in vector.Values)
        {
            if ((uint)pair.Key >= (uint)_model.Classifier.Weights.Length)
                continue;

            var contribution = _model.Classifier.Weights[pair.Key] * pair.Value;
            var label = vector.Labels.TryGetValue(pair.Key, out var found)
                ? found
                : $"feature#{pair.Key}";
            list.Add(new FeatureExplanation(label, contribution));
        }

        return list
            .OrderByDescending(x => Math.Abs(x.Contribution))
            .Take(take)
            .ToArray();
    }
}
