"""Charset recovery for real MIME mail; frozen parsers remain unchanged."""
from email import policy
from email.parser import BytesParser
from train_v47_phishing_aware import canonical_fields


def decode_payload(payload, charset=None):
    if not isinstance(payload, bytes):
        return str(payload or '')
    if charset:
        try:
            return payload.decode(charset, 'strict')
        except (LookupError, UnicodeError):
            pass
    for encoding in ('utf-8', 'iso2022_jp'):
        try:
            return payload.decode(encoding, 'strict')
        except UnicodeError:
            pass
    try:
        recovered = payload.decode('gb18030', 'strict')
        letters = [c for c in recovered if c.isalpha()]
        if sum('\u3400' <= c <= '\u9fff' for c in letters) / max(1, len(letters)) > .25:
            return recovered
    except UnicodeError:
        pass
    return payload.decode('cp1252', 'replace')


def from_eml(raw):
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    parts = []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        if part.get_content_type() not in ('text/plain', 'text/html'):
            continue
        parts.append(decode_payload(part.get_payload(decode=True), part.get_content_charset()))
    subject = str(msg.get('subject') or '')
    original_subject = next((v for k, v in msg.raw_items() if k.lower() == 'subject'), '')
    if any('\udc80' <= c <= '\udcff' for c in original_subject):
        subject = decode_payload(original_subject.encode('ascii', 'surrogateescape'), msg.get_content_charset())
    return canonical_fields(subject, '\n'.join(parts), str(msg.get('from') or ''))
