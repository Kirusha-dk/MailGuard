using MailGuard.Core.Features;

namespace MailGuard.Core.Ml;

public sealed record TrainingExample(
    SparseFeatureVector Features,
    bool IsSpam,
    bool RspamdAlreadySpam = false);

public sealed class LogisticRegressionModel
{
    public int Dimension => Weights.Length;
    public double[] Weights { get; }
    public double Bias { get; set; }

    public LogisticRegressionModel(int dimension)
    {
        Weights = new double[dimension];
    }

    public LogisticRegressionModel(double[] weights, double bias)
    {
        Weights = weights;
        Bias = bias;
    }

    public double PredictProbability(SparseFeatureVector features)
    {
        var z = Bias;

        foreach (var pair in features.Values)
        {
            if ((uint)pair.Key < (uint)Weights.Length)
                z += Weights[pair.Key] * pair.Value;
        }

        return Sigmoid(z);
    }

    public static LogisticRegressionModel Train(
        IReadOnlyList<TrainingExample> examples,
        int dimension,
        int epochs = 10,
        double learningRate = 0.06,
        double l2 = 0.00001,
        int seed = 1337,
        Action<int, double>? epochCompleted = null)
    {
        if (examples.Count == 0)
            throw new ArgumentException("Training set is empty.");

        var model = new LogisticRegressionModel(dimension);
        var indices = Enumerable.Range(0, examples.Count).ToArray();
        var random = new Random(seed);

        var spamCount = Math.Max(1, examples.Count(x => x.IsSpam));
        var hamCount = Math.Max(1, examples.Count - spamCount);
        var spamWeight = examples.Count / (2.0 * spamCount);
        var hamWeight = examples.Count / (2.0 * hamCount);

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(indices, random);
            var lr = learningRate / Math.Sqrt(1.0 + epoch * 0.35);
            var totalLoss = 0.0;

            foreach (var index in indices)
            {
                var sample = examples[index];
                var target = sample.IsSpam ? 1.0 : 0.0;
                var classWeight = sample.IsSpam ? spamWeight : hamWeight;

                // The local model exists mainly to recover spam that Rspamd missed.
                // Give those residual errors more influence while threshold tuning
                // still constrains false positives on validation mail.
                if (sample.IsSpam && !sample.RspamdAlreadySpam)
                    classWeight *= 2.5;

                var probability = model.PredictProbability(sample.Features);
                var error = (probability - target) * classWeight;

                foreach (var pair in sample.Features.Values)
                {
                    var w = model.Weights[pair.Key];
                    var gradient = error * pair.Value + l2 * w;
                    model.Weights[pair.Key] = w - lr * gradient;
                }

                model.Bias -= lr * error;

                var p = Math.Clamp(probability, 1e-8, 1 - 1e-8);
                totalLoss += -classWeight * (
                    target * Math.Log(p)
                    + (1 - target) * Math.Log(1 - p));
            }

            epochCompleted?.Invoke(epoch + 1, totalLoss / examples.Count);
        }

        return model;
    }

    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            var z = Math.Exp(-x);
            return 1.0 / (1.0 + z);
        }

        var e = Math.Exp(x);
        return e / (1.0 + e);
    }

    private static void Shuffle(int[] array, Random random)
    {
        for (var i = array.Length - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
