using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using MailGuard.Core.Domain;

namespace MailGuard.Core.Email;

public sealed class EmailParser
{
    private static readonly Encoding ByteEncoding = Encoding.Latin1;

    private static readonly Regex EncodedWordRegex = new(
        @"=\?([^?\s]+)\?([bBqQ])\?([^?]*)\?=",
        RegexOptions.Compiled);

    private static readonly Regex BoundaryRegex = new(
        @"boundary\s*=\s*(?:""([^""]+)""|([^;\s]+))",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex CharsetRegex = new(
        @"charset\s*=\s*(?:""([^""]+)""|([^;\s]+))",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex HtmlTagRegex = new(
        @"<[^>]+>",
        RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly Regex UrlRegex = new(
        @"https?://[^\s<>'""]+",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public ParsedEmail Parse(byte[] rawMessage)
    {
        ArgumentNullException.ThrowIfNull(rawMessage);

        var entity = SplitEntity(rawMessage);
        var headers = ParseHeaders(entity.Headers);
        var bodyText = new StringBuilder();
        var attachmentCount = 0;
        var hasHtml = false;

        ParseBody(
            headers,
            entity.Body,
            bodyText,
            ref attachmentCount,
            ref hasHtml,
            depth: 0);

        var text = NormalizeText(bodyText.ToString());

        return new ParsedEmail(
            Subject: DecodeHeader(Get(headers, "Subject")),
            From: DecodeHeader(Get(headers, "From")),
            To: DecodeHeader(Get(headers, "To")),
            Text: text,
            Headers: headers,
            UrlCount: UrlRegex.Matches(text).Count,
            AttachmentCount: attachmentCount,
            HasHtml: hasHtml,
            RawLength: rawMessage.Length);
    }

    private static void ParseBody(
        IReadOnlyDictionary<string, string> headers,
        byte[] body,
        StringBuilder output,
        ref int attachmentCount,
        ref bool hasHtml,
        int depth)
    {
        if (depth > 20)
            return;

        var contentType = Get(headers, "Content-Type");
        var disposition = Get(headers, "Content-Disposition");
        var transferEncoding = Get(headers, "Content-Transfer-Encoding");

        if (LooksLikeAttachment(contentType, disposition))
            attachmentCount++;

        if (contentType.StartsWith("multipart/", StringComparison.OrdinalIgnoreCase))
        {
            var boundary = GetParameter(contentType, BoundaryRegex);
            if (string.IsNullOrWhiteSpace(boundary))
                return;

            foreach (var part in SplitMultipart(body, boundary))
            {
                var entity = SplitEntity(part);
                var partHeaders = ParseHeaders(entity.Headers);
                ParseBody(
                    partHeaders,
                    entity.Body,
                    output,
                    ref attachmentCount,
                    ref hasHtml,
                    depth + 1);
            }

            return;
        }

        if (contentType.StartsWith("message/rfc822", StringComparison.OrdinalIgnoreCase))
        {
            var nested = DecodeTransfer(body, transferEncoding);
            var entity = SplitEntity(nested);
            var nestedHeaders = ParseHeaders(entity.Headers);
            ParseBody(
                nestedHeaders,
                entity.Body,
                output,
                ref attachmentCount,
                ref hasHtml,
                depth + 1);
            return;
        }

        var isHtml = contentType.StartsWith("text/html", StringComparison.OrdinalIgnoreCase);
        var isText = string.IsNullOrWhiteSpace(contentType)
                     || contentType.StartsWith("text/", StringComparison.OrdinalIgnoreCase);

        if (!isText)
            return;

        var decodedBytes = DecodeTransfer(body, transferEncoding);
        var charset = GetParameter(contentType, CharsetRegex);
        var text = DecodeText(decodedBytes, charset);

        if (isHtml)
        {
            hasHtml = true;
            text = HtmlToText(text);
        }

        if (!string.IsNullOrWhiteSpace(text))
        {
            if (output.Length > 0)
                output.AppendLine();
            output.Append(text);
        }
    }

    private static EntityParts SplitEntity(byte[] raw)
    {
        var text = ByteEncoding.GetString(raw);
        var index = text.IndexOf("\r\n\r\n", StringComparison.Ordinal);
        var separatorLength = 4;

        if (index < 0)
        {
            index = text.IndexOf("\n\n", StringComparison.Ordinal);
            separatorLength = 2;
        }

        if (index < 0)
            return new EntityParts(text, Array.Empty<byte>());

        var headers = text[..index];
        var bodyText = text[(index + separatorLength)..];
        return new EntityParts(headers, ByteEncoding.GetBytes(bodyText));
    }

    private static Dictionary<string, string> ParseHeaders(string rawHeaders)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        string? currentName = null;
        var currentValue = new StringBuilder();

        void Flush()
        {
            if (string.IsNullOrWhiteSpace(currentName))
                return;

            var value = currentValue.ToString().Trim();
            if (result.TryGetValue(currentName, out var existing))
                result[currentName] = existing + ", " + value;
            else
                result[currentName] = value;
        }

        using var reader = new StringReader(rawHeaders);
        while (reader.ReadLine() is { } line)
        {
            if ((line.StartsWith(' ') || line.StartsWith('\t')) && currentName is not null)
            {
                currentValue.Append(' ').Append(line.Trim());
                continue;
            }

            Flush();
            currentName = null;
            currentValue.Clear();

            var colon = line.IndexOf(':');
            if (colon <= 0)
                continue;

            currentName = line[..colon].Trim();
            currentValue.Append(line[(colon + 1)..].Trim());
        }

        Flush();
        return result;
    }

    private static IEnumerable<byte[]> SplitMultipart(byte[] body, string boundary)
    {
        var text = ByteEncoding.GetString(body);
        var marker = "--" + boundary;
        var chunks = text.Split(marker, StringSplitOptions.None);

        foreach (var rawChunk in chunks.Skip(1))
        {
            var chunk = rawChunk;

            if (chunk.StartsWith("--", StringComparison.Ordinal))
                yield break;

            chunk = chunk.TrimStart('\r', '\n');
            chunk = chunk.TrimEnd('\r', '\n');

            if (chunk.Length == 0)
                continue;

            yield return ByteEncoding.GetBytes(chunk);
        }
    }

    private static byte[] DecodeTransfer(byte[] body, string transferEncoding)
    {
        var encoding = transferEncoding.Trim().ToLowerInvariant();

        if (encoding == "base64")
        {
            try
            {
                var compact = Regex.Replace(ByteEncoding.GetString(body), @"\s+", "");
                return Convert.FromBase64String(compact);
            }
            catch
            {
                return body;
            }
        }

        if (encoding == "quoted-printable")
            return DecodeQuotedPrintable(body);

        return body;
    }

    private static byte[] DecodeQuotedPrintable(byte[] input)
    {
        using var output = new MemoryStream(input.Length);

        for (var i = 0; i < input.Length; i++)
        {
            if (input[i] != (byte)'=')
            {
                output.WriteByte(input[i]);
                continue;
            }

            if (i + 1 < input.Length && input[i + 1] == (byte)'\n')
            {
                i += 1;
                continue;
            }

            if (i + 2 < input.Length
                && input[i + 1] == (byte)'\r'
                && input[i + 2] == (byte)'\n')
            {
                i += 2;
                continue;
            }

            if (i + 2 < input.Length
                && TryHex(input[i + 1], out var high)
                && TryHex(input[i + 2], out var low))
            {
                output.WriteByte((byte)((high << 4) | low));
                i += 2;
                continue;
            }

            output.WriteByte(input[i]);
        }

        return output.ToArray();
    }

    private static bool TryHex(byte value, out int digit)
    {
        if (value is >= (byte)'0' and <= (byte)'9')
        {
            digit = value - (byte)'0';
            return true;
        }

        if (value is >= (byte)'A' and <= (byte)'F')
        {
            digit = value - (byte)'A' + 10;
            return true;
        }

        if (value is >= (byte)'a' and <= (byte)'f')
        {
            digit = value - (byte)'a' + 10;
            return true;
        }

        digit = 0;
        return false;
    }

    private static string DecodeHeader(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        return EncodedWordRegex.Replace(value, match =>
        {
            try
            {
                var charset = match.Groups[1].Value;
                var mode = match.Groups[2].Value;
                var payload = match.Groups[3].Value;

                byte[] bytes;
                if (mode.Equals("B", StringComparison.OrdinalIgnoreCase))
                {
                    bytes = Convert.FromBase64String(payload);
                }
                else
                {
                    payload = payload.Replace('_', ' ');
                    bytes = DecodeQuotedPrintable(ByteEncoding.GetBytes(payload));
                }

                return DecodeText(bytes, charset);
            }
            catch
            {
                return match.Value;
            }
        });
    }

    private static string DecodeText(byte[] bytes, string? charset)
    {
        if (bytes.Length == 0)
            return string.Empty;

        try
        {
            if (string.IsNullOrWhiteSpace(charset))
                return Encoding.UTF8.GetString(bytes);

            var normalized = charset.Trim().Trim('"').ToLowerInvariant();
            return normalized switch
            {
                "utf-8" or "utf8" => Encoding.UTF8.GetString(bytes),
                "us-ascii" or "ascii" => Encoding.ASCII.GetString(bytes),
                "iso-8859-1" or "latin1" or "latin-1" => Encoding.Latin1.GetString(bytes),
                "windows-1252" or "cp1252" => Encoding.Latin1.GetString(bytes),
                _ => Encoding.UTF8.GetString(bytes)
            };
        }
        catch
        {
            return Encoding.UTF8.GetString(bytes);
        }
    }

    private static string HtmlToText(string html)
    {
        var withoutScripts = Regex.Replace(
            html,
            @"<(script|style)\b[^>]*>.*?</\1>",
            " ",
            RegexOptions.IgnoreCase | RegexOptions.Singleline);

        var withoutTags = HtmlTagRegex.Replace(withoutScripts, " ");
        return WebUtility.HtmlDecode(withoutTags);
    }

    private static string NormalizeText(string text)
    {
        text = text.Replace("\r\n", "\n").Replace('\r', '\n');
        text = Regex.Replace(text, @"[\t ]+", " ");
        text = Regex.Replace(text, @"\n{3,}", "\n\n");
        return text.Trim();
    }

    private static bool LooksLikeAttachment(string contentType, string disposition)
    {
        if (disposition.Contains("attachment", StringComparison.OrdinalIgnoreCase))
            return true;

        return disposition.Contains("filename=", StringComparison.OrdinalIgnoreCase)
               || contentType.Contains("name=", StringComparison.OrdinalIgnoreCase);
    }

    private static string GetParameter(string header, Regex regex)
    {
        if (string.IsNullOrWhiteSpace(header))
            return string.Empty;

        var match = regex.Match(header);
        if (!match.Success)
            return string.Empty;

        return match.Groups[1].Success
            ? match.Groups[1].Value
            : match.Groups[2].Value;
    }

    private static string Get(IReadOnlyDictionary<string, string> headers, string name) =>
        headers.TryGetValue(name, out var value) ? value : string.Empty;

    private readonly record struct EntityParts(string Headers, byte[] Body);
}
