"""MEXC's spot websocket messages, decoded from the ``.proto`` files MEXC publishes.

The files beside this module are ``mexcdevelop/websocket-proto`` at commit
``0c9c4f35dd0fadc3a46a350e909a93379d81e811`` (Apache License 2.0, ``LICENSE`` here),
unchanged. MEXC's v3 websocket sends every push as one ``PushDataV3ApiWrapper``: the
channel, the symbol, ``createTime`` and ``sendTime`` in ms, and a ``oneof body`` that
holds the channel's own message.

No protobuf library is needed. :func:`schema` reads the field names, numbers and types of
every message from the files, and :func:`decode` walks the wire format: each field is a
key, a varint of ``number << 3 | wire type``, then a varint (type 0), eight bytes (1), a
length and that many bytes (2) or four bytes (5). A string, a bytes field and a message
are length-delimited; a repeated number may be packed into one length-delimited field.
A field the schema does not name is skipped by its wire type, as protobuf does, so a
field MEXC adds later cannot break the decode. :func:`encode` writes the same format and
exists for tests, which build frames by hand.
"""

from __future__ import annotations

import functools
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "0c9c4f35dd0fadc3a46a350e909a93379d81e811"
WRAPPER = "PushDataV3ApiWrapper"

_VARINT = {"int32", "int64", "uint32", "uint64", "bool", "enum", "sint32", "sint64"}
_FIXED64 = {"fixed64", "sfixed64", "double"}
_FIXED32 = {"fixed32", "sfixed32", "float"}
_FIELD = re.compile(r"^(repeated|optional)?\s*([\w.]+)\s+(\w+)\s*=\s*(\d+)\s*;")


class DecodeError(ValueError):
    """Bytes that are not a well-formed message of the schema."""


@functools.cache
def schema() -> dict[str, dict[int, tuple[str, str, bool, str | None]]]:
    """Every message of the vendored files: ``{message: {number: (name, type, repeated,
    oneof)}}``; ``oneof`` names the ``oneof`` group a field belongs to, or None."""
    out: dict[str, dict] = {}
    for path in sorted(HERE.glob("*.proto")):
        text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        message, oneof, depth = None, None, 0
        for line in (x.strip() for x in text.splitlines()):
            if not line:
                continue
            m = re.match(r"message\s+(\w+)\s*\{", line)
            if m and depth == 0:
                message, depth = m.group(1), 1
                out[message] = {}
                continue
            m = re.match(r"oneof\s+(\w+)\s*\{", line)
            if m and message:
                oneof, depth = m.group(1), depth + 1
                continue
            if line.startswith("}"):
                depth -= 1
                if depth == 1:
                    oneof = None
                elif depth == 0:
                    message = None
                continue
            m = _FIELD.match(line)
            if m and message:
                label, kind, name, number = m.groups()
                out[message][int(number)] = (name, kind, label == "repeated", oneof)
    return out


def _varint(data: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        if i >= len(data):
            raise DecodeError("a varint runs past the end")
        b = data[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise DecodeError("a varint longer than ten bytes")


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def _scalar(kind: str, value: int) -> int | bool:
    if kind == "bool":
        return bool(value)
    if kind in ("int32", "enum"):
        return _signed(value & 0xFFFFFFFF, 32) if value < 1 << 32 else _signed(value, 64)
    if kind == "int64":
        return _signed(value, 64)
    if kind in ("sint32", "sint64"):
        return (value >> 1) ^ -(value & 1)
    return value


def decode(message: str, data: bytes) -> dict:
    """``data`` as ``message``: a dict of the fields present, a repeated field a list, a
    message field a dict; for a field of a ``oneof``, ``"_" + group`` holds its name.

    Examples
    --------
    >>> d = decode("PublicAggreBookTickerV3Api", bytes.fromhex("0a04302e3135" "3007"))
    >>> d["bidPrice"], d["lastOrderCreateTime"]
    ('0.15', 7)
    """
    fields = schema().get(message)
    if fields is None:
        raise DecodeError(f"no message {message!r} in the vendored files")
    out: dict = {}
    i, n = 0, len(data)
    while i < n:
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        spec = fields.get(number)
        if wire == 0:
            raw, i = _varint(data, i)
            if spec is not None and spec[1] not in _VARINT:
                raise DecodeError(f"a varint for {spec[0]}, a field of type {spec[1]}")
            value = None if spec is None else _scalar(spec[1], raw)
        elif wire == 1:
            if i + 8 > n:
                raise DecodeError("a 64-bit field runs past the end")
            chunk, i = data[i : i + 8], i + 8
            value = None if spec is None else _fixed(spec[1], chunk)
        elif wire == 5:
            if i + 4 > n:
                raise DecodeError("a 32-bit field runs past the end")
            chunk, i = data[i : i + 4], i + 4
            value = None if spec is None else _fixed(spec[1], chunk)
        elif wire == 2:
            length, i = _varint(data, i)
            if i + length > n:
                raise DecodeError("a length-delimited field runs past the end")
            chunk, i = data[i : i + length], i + length
            if spec is None:
                continue
            value = _delimited(spec, chunk)
        else:
            raise DecodeError(f"wire type {wire} is not one proto3 writes")
        if spec is None:
            continue
        name, kind, repeated, group = spec
        if repeated:
            out.setdefault(name, []).extend(value if isinstance(value, list) else [value])
        else:
            out[name] = value
        if group is not None:
            out["_" + group] = name
    return out


def _fixed(kind: str, chunk: bytes):
    fmt = {"fixed64": "<Q", "sfixed64": "<q", "double": "<d", "fixed32": "<I",
           "sfixed32": "<i", "float": "<f"}.get(kind)  # fmt: skip
    if fmt is None:
        raise DecodeError(f"a fixed-width value for a field of type {kind}")
    return struct.unpack(fmt, chunk)[0]


def _delimited(spec, chunk: bytes):
    _, kind, repeated, _ = spec
    if kind == "string":
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodeError("a string field that is not UTF-8") from exc
    if kind == "bytes":
        return bytes(chunk)
    if kind in _VARINT and repeated:  # packed
        values, j = [], 0
        while j < len(chunk):
            raw, j = _varint(chunk, j)
            values.append(_scalar(kind, raw))
        return values
    if (kind in _FIXED32 or kind in _FIXED64) and repeated:
        width = 4 if kind in _FIXED32 else 8
        return [_fixed(kind, chunk[j : j + width]) for j in range(0, len(chunk), width)]
    if kind in schema():
        return decode(kind, chunk)
    raise DecodeError(f"a length-delimited value for a field of type {kind}")


def _put_varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode(message: str, values: dict) -> bytes:
    """``values`` as ``message`` in the wire format, fields in number order, for tests."""
    fields = schema()[message]
    out = bytearray()
    for number in sorted(fields):
        name, kind, repeated, _ = fields[number]
        if name not in values or values[name] is None:
            continue
        for v in values[name] if repeated else [values[name]]:
            if kind == "string" or kind == "bytes" or kind in schema():
                body = (v.encode() if kind == "string" else bytes(v) if kind == "bytes"
                        else encode(kind, v))  # fmt: skip
                out += _put_varint(number << 3 | 2) + _put_varint(len(body)) + body
            elif kind in _VARINT:
                raw = int(v)
                if kind in ("sint32", "sint64"):
                    raw = (raw << 1) ^ (raw >> 63)
                out += _put_varint(number << 3) + _put_varint(raw)
            else:
                raise ValueError(f"encode does not write fields of type {kind}")
    return bytes(out)


def wrapper(data: bytes) -> dict:
    """One push frame: :func:`decode` of ``PushDataV3ApiWrapper``, with ``body`` the name
    of the ``oneof`` member it carries (None when none)."""
    out = decode(WRAPPER, data)
    out["body"] = out.pop("_body", None)
    return out
