namespace MailGuard.Core.Data;

public sealed class LeakageIndex
{
    private readonly HashSet<string> _exact;
    private readonly Dictionary<ushort, List<ulong>>[] _buckets;

    public LeakageIndex(IEnumerable<string> exactHashes, IEnumerable<ulong> simHashes)
    {
        _exact = new HashSet<string>(exactHashes, StringComparer.OrdinalIgnoreCase);
        _buckets = Enumerable.Range(0, 4)
            .Select(_ => new Dictionary<ushort, List<ulong>>())
            .ToArray();

        foreach (var hash in simHashes.Distinct())
        {
            for (var part = 0; part < 4; part++)
            {
                var key = (ushort)((hash >> (part * 16)) & 0xffff);
                if (!_buckets[part].TryGetValue(key, out var list))
                {
                    list = new List<ulong>();
                    _buckets[part][key] = list;
                }
                list.Add(hash);
            }
        }
    }

    public bool IsExactDuplicate(MessageFingerprint fingerprint) =>
        _exact.Contains(fingerprint.ExactSha256);

    public bool IsNearDuplicate(MessageFingerprint fingerprint, int maxDistance = 2)
    {
        var candidates = new HashSet<ulong>();
        for (var part = 0; part < 4; part++)
        {
            var key = (ushort)((fingerprint.SimHash >> (part * 16)) & 0xffff);
            if (_buckets[part].TryGetValue(key, out var list))
            {
                foreach (var item in list)
                    candidates.Add(item);
            }
        }

        foreach (var candidate in candidates)
        {
            if (DatasetFingerprint.HammingDistance(candidate, fingerprint.SimHash) <= maxDistance)
                return true;
        }

        return false;
    }
}
