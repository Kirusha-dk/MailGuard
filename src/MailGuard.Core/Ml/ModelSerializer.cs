using System.Text;

namespace MailGuard.Core.Ml;

public static class ModelSerializer
{
    private const string Magic = "MAILGUARD043";

    public static void Save(string path, MailGuardModel model)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        Validate(model);
        var temp = Path.GetFullPath(path) + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            Write(temp, model);
            File.Move(temp, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    private static void Write(string temp, MailGuardModel model)
    {
        using var stream = File.Create(temp);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);

        writer.Write(Magic);
        writer.Write(model.Version);
        writer.Write(model.Classifier.Dimension);
        writer.Write(model.Classifier.Bias);
        writer.Write(model.SpamThreshold);
        writer.Write(model.HamThreshold);
        writer.Write(model.ProtectRspamdSpam);

        foreach (var weight in model.Classifier.Weights)
            writer.Write((float)weight);

        writer.Write(model.TrainingExactHashes.Count);
        foreach (var hash in model.TrainingExactHashes)
            writer.Write(hash);

        writer.Write(model.TrainingSimHashes.Count);
        foreach (var hash in model.TrainingSimHashes)
            writer.Write(hash);
    }

    public static MailGuardModel Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

        var magic = reader.ReadString();
        if (magic != Magic)
        {
            throw new InvalidDataException(
                "Expected a compatible MailGuard 0.4.x .mg package. Python v80/v82 models require their own runtime; do not restart training to replace missing weights.");
        }

        var version = reader.ReadString();
        var dimension = reader.ReadInt32();
        var bias = reader.ReadDouble();
        var spamThreshold = reader.ReadDouble();
        var hamThreshold = reader.ReadDouble();
        var protectRspamd = reader.ReadBoolean();

        if (dimension <= 1024 || dimension > (1 << 24) || (dimension & (dimension - 1)) != 0 ||
            (long)dimension * sizeof(float) > stream.Length - stream.Position)
            throw new InvalidDataException("Invalid model dimension or truncated weights.");
        var weights = new double[dimension];
        for (var i = 0; i < dimension; i++)
            weights[i] = reader.ReadSingle();

        var model = new MailGuardModel(new LogisticRegressionModel(weights, bias))
        {
            Version = version,
            SpamThreshold = spamThreshold,
            HamThreshold = hamThreshold,
            ProtectRspamdSpam = protectRspamd
        };

        var exactCount = reader.ReadInt32();
        if (exactCount < 0 || exactCount > stream.Length - stream.Position)
            throw new InvalidDataException("Invalid training hash count.");
        for (var i = 0; i < exactCount; i++)
            model.TrainingExactHashes.Add(reader.ReadString());

        var simCount = reader.ReadInt32();
        if (simCount < 0 || (long)simCount * sizeof(ulong) > stream.Length - stream.Position)
            throw new InvalidDataException("Invalid similarity hash count.");
        for (var i = 0; i < simCount; i++)
            model.TrainingSimHashes.Add(reader.ReadUInt64());

        if (stream.Position != stream.Length)
            throw new InvalidDataException("Unexpected data after the model.");
        Validate(model);
        return model;
    }

    private static void Validate(MailGuardModel model)
    {
        var dimension = model.Classifier.Dimension;
        if (dimension <= 1024 || dimension > (1 << 24) || (dimension & (dimension - 1)) != 0 ||
            !double.IsFinite(model.Classifier.Bias) ||
            model.Classifier.Weights.Any(w => !float.IsFinite((float)w)) ||
            !double.IsFinite(model.SpamThreshold) || !double.IsFinite(model.HamThreshold) ||
            model.HamThreshold < 0 || model.HamThreshold > 1 || model.SpamThreshold < model.HamThreshold)
            throw new InvalidDataException("Invalid model parameters or thresholds.");
    }
}
