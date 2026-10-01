using MailGuard.Core.Domain;

namespace MailGuard.Core.Rspamd;

public interface IRspamdScanner
{
    Task<RspamdScanResult> ScanAsync(
        byte[] rawMessage,
        CancellationToken cancellationToken = default);
}
