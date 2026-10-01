using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Rspamd;

public sealed class CachedRspamdScanner : IRspamdScanner
{
    private readonly IRspamdScanner _inner;
    private readonly string _cacheDirectory;
    private readonly string _cacheNamespace;
    private readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web);

    public CachedRspamdScanner(
        IRspamdScanner inner,
        string cacheDirectory,
        string cacheNamespace = "default")
    {
        _inner = inner;
        _cacheDirectory = cacheDirectory;
        _cacheNamespace = cacheNamespace;
        Directory.CreateDirectory(_cacheDirectory);
    }

    public async Task<RspamdScanResult> ScanAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default)
    {
        var key = ComputeKey(rawMessage, _cacheNamespace);
        var path = Path.Combine(_cacheDirectory, key[..2], key + ".json");

        if (File.Exists(path))
        {
            try
            {
                await using var read = File.OpenRead(path);
                var cached = await JsonSerializer.DeserializeAsync<RspamdScanResult>(
                    read, _json, cancellationToken);
                if (cached is not null)
                    return cached;
            }
            catch
            {
                // Corrupted cache entries are ignored and rebuilt.
            }
        }

        var result = await _inner.ScanAsync(rawMessage, cancellationToken);

        var directory = Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(directory);
        var temp = path + ".tmp-" + Guid.NewGuid().ToString("N");

        await using (var write = File.Create(temp))
        {
            await JsonSerializer.SerializeAsync(
                write, result, _json, cancellationToken);
        }

        File.Move(temp, path, true);
        return result;
    }

    private static string ComputeKey(byte[] rawMessage, string cacheNamespace)
    {
        using var sha = SHA256.Create();
        var prefix = Encoding.UTF8.GetBytes(cacheNamespace + "\n");
        var data = new byte[prefix.Length + rawMessage.Length];
        Buffer.BlockCopy(prefix, 0, data, 0, prefix.Length);
        Buffer.BlockCopy(rawMessage, 0, data, prefix.Length, rawMessage.Length);
        return Convert.ToHexString(sha.ComputeHash(data)).ToLowerInvariant();
    }
}
