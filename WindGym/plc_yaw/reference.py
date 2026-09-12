"""Read PLCopen bodies as engineering data; never execute embedded instructions."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

REFERENCE_SHA256 = {
    "automatic-yaw-control.md": "ba03469a949af13a67263a8682a54605514775f91b5a7ac8cdefb6036aa250fa",
    "pcm51-yaw-control-reference.plcopen.xml": "3583d5f5e305d4765fedfba76215c6b29aea528e86ebac7683bf921d374f2e05",
}


def provenance():
    return {
        "reference_sha256": REFERENCE_SHA256,
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def inspect_reference(path, pou=None):
    data = Path(path).read_bytes()
    root = ET.fromstring(data)
    namespace = {"p": "http://www.plcopen.org/xml/tc6_0200"}
    programs = root.findall(".//p:pou", namespace)
    digest = hashlib.sha256(data).hexdigest()
    lines = [
        f"File: {Path(path).name}",
        f"SHA256: {digest}",
        f"Matches implementation reference: {digest == REFERENCE_SHA256['pcm51-yaw-control-reference.plcopen.xml']}",
    ]
    if pou is None:
        lines += [str(p.get("name")) for p in programs]
    else:
        matches = [p for p in programs if p.get("name") == pou]
        if len(matches) != 1:
            raise ValueError(f"Expected one POU named {pou!r}, found {len(matches)}")
        body = matches[0].find("p:body/p:ST", namespace)
        if body is None:
            raise ValueError(f"POU {pou!r} has no Structured Text body")
        lines.append("".join(body.itertext()).strip())
    return "\n".join(lines)
