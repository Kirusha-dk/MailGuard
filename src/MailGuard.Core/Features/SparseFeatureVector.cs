namespace MailGuard.Core.Features;

public sealed class SparseFeatureVector
{
    public Dictionary<int, double> Values { get; } = new();
    public Dictionary<int, string> Labels { get; } = new();

    public void Add(int index, double value, string label)
    {
        if (value == 0)
            return;

        Values[index] = Values.TryGetValue(index, out var old)
            ? old + value
            : value;

        if (!Labels.ContainsKey(index))
            Labels[index] = label;
    }
}
