using System.Buffers;

namespace MailGuard.Core.Email;

public static class MessageReader
{
    public const int DefaultMaxBytes = 20_000_000;

    public static async Task<byte[]> ReadAsync(Stream source, int maxBytes = DefaultMaxBytes,
        CancellationToken cancellationToken = default)
    {
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes));
        var buffer = ArrayPool<byte>.Shared.Rent(81920);
        try
        {
            using var output = new MemoryStream();
            int count;
            while ((count = await source.ReadAsync(buffer.AsMemory(), cancellationToken)) != 0)
            {
                if (count > maxBytes - output.Length)
                    throw new MessageTooLargeException(maxBytes);
                await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
            }
            cancellationToken.ThrowIfCancellationRequested();
            return output.ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }
}

public sealed class MessageTooLargeException(int maxBytes)
    : IOException($"Message exceeds the configured limit of {maxBytes} bytes.");
