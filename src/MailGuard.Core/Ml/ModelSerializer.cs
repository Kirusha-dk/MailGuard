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

        using var stream = File.Create(path);
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
                "This model is incompatible with the current MailGuard 0.4.x format. Retrain it with the current version.");
        }

        var version = reader.ReadString();
        var dimension = reader.ReadInt32();
        var bias = reader.ReadDouble();
        var spamThreshold = reader.ReadDouble();
        var hamThreshold = reader.ReadDouble();
        var protectRspamd = reader.ReadBoolean();

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
        for (var i = 0; i < exactCount; i++)
            model.TrainingExactHashes.Add(reader.ReadString());

        var simCount = reader.ReadInt32();
        for (var i = 0; i < simCount; i++)
            model.TrainingSimHashes.Add(reader.ReadUInt64());

        return model;
    }
}
