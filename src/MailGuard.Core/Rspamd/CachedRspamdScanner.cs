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
        cancellationToken.ThrowIfCancellationRequested();
        var key = ComputeKey(rawMessage, _cacheNamespace);
        var path = Path.Combine(_cacheDirectory, key[..2], key + ".json");

        if (File.Exists(path))
        {
            try
            {
                await using var read = File.OpenRead(path);
                var cached = await JsonSerializer.DeserializeAsync<RspamdScanResult>(
                    read, _json, cancellationToken);
                if (cached is not null && !string.IsNullOrWhiteSpace(cached.Action)
                    && double.IsFinite(cached.Score) && double.IsFinite(cached.RequiredScore)
                    && cached.Symbols is not null && cached.Symbols.All(s => s is not null
                        && s.Name is not null && double.IsFinite(s.Score) && s.Options is not null))
                    return cached;
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                // Corrupted cache entries are ignored and rebuilt.
            }
        }

        var result = await _inner.ScanAsync(rawMessage, cancellationToken);

        var temp = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await using (var write = File.Create(temp))
            {
                await JsonSerializer.SerializeAsync(write, result, _json, cancellationToken);
            }
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temp, path, true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // A cache failure must not discard a successful scan.
        }
        finally
        {
            try { File.Delete(temp); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
        return result;
    }

    private static string ComputeKey(byte[] rawMessage, string cacheNamespace)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        sha.AppendData(Encoding.UTF8.GetBytes(cacheNamespace + "\n"));
        sha.AppendData(rawMessage);
        return Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
    }
}
