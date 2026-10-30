namespace MailGuard.Core.Rspamd;

public sealed class RspamdException : Exception
{
    public RspamdException(string message, Exception? inner = null) : base(message, inner) { }
}
