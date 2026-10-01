using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Data;

public sealed record MessageFingerprint(string ExactSha256, ulong SimHash);

public static class DatasetFingerprint
{
    private static readonly Regex TokenRegex = new(
        @"[\p{L}\p{Nd}][\p{L}\p{Nd}_\-.@]{1,48}",
        RegexOptions.Compiled);

    public static MessageFingerprint Compute(ParsedEmail email)
    {
        var normalized = Normalize(email.Subject + "\n" + email.From + "\n" + email.Text);
        var exact = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(normalized)))
            .ToLowerInvariant();
        var sim = ComputeSimHash(normalized);
        return new MessageFingerprint(exact, sim);
    }

    public static int HammingDistance(ulong a, ulong b) =>
        BitOperations.PopCount(a ^ b);

    private static string Normalize(string text)
    {
        text = text.ToLowerInvariant();
        text = Regex.Replace(text, @"https?://\S+", " <url> ");
        text = Regex.Replace(text, @"\b\d{2,}\b", " <num> ");
        text = Regex.Replace(text, @"\s+", " ");
        return text.Trim();
    }

    private static ulong ComputeSimHash(string text)
    {
        var votes = new int[64];
        var tokens = TokenRegex.Matches(text)
            .Cast<Match>()
            .Select(x => x.Value)
            .Take(8000);

        var count = 0;
        foreach (var token in tokens)
        {
            count++;
            var h = Fnv1A64(token);
            for (var bit = 0; bit < 64; bit++)
                votes[bit] += ((h >> bit) & 1UL) != 0 ? 1 : -1;
        }

        if (count == 0)
            return 0;

        ulong result = 0;
        for (var bit = 0; bit < 64; bit++)
        {
            if (votes[bit] >= 0)
                result |= 1UL << bit;
        }
        return result;
    }

    private static ulong Fnv1A64(string value)
    {
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (var c in value)
            {
                hash ^= c;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }
}
