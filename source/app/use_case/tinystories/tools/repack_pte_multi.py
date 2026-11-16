# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Replace every delegate payload in a multi-delegate PTE."""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'dependencies/executorch'))
import executorch  # noqa: E402

executorch.__path__ = [str(ROOT / 'dependencies/executorch'), *executorch.__path__]
from executorch.exir._serialize._program import (  # noqa: E402
    PTEFile, deserialize_pte_binary, serialize_pte_binary,
)


def main():
    """Rebuild a PTE with a payload directory containing payload_XX/vela_payload.bin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('payload_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pte = deserialize_pte_binary(args.source.read_bytes())
    count = len(pte.program.backend_delegate_data)
    for index in range(count):
        payload = (args.payload_dir / f'payload_{index:02d}' / 'vela_payload.bin').read_bytes()
        pte.program.backend_delegate_data[index].data = payload
        print(f'delegate {index}: {len(payload)} bytes '
              f'{hashlib.sha256(payload).hexdigest()[:16]}')
    serialized = bytes(serialize_pte_binary(PTEFile(
        program=pte.program,
        mutable_data=pte.mutable_data,
        named_data=pte.named_data,
    )))
    args.output.write_bytes(serialized)
    check = deserialize_pte_binary(args.output.read_bytes())
    assert len(check.program.backend_delegate_data) == count
    for index in range(count):
        expected = (args.payload_dir / f'payload_{index:02d}' / 'vela_payload.bin').read_bytes()
        assert check.program.backend_delegate_data[index].data == expected
    print('plans', len(check.program.execution_plan), 'delegates', count,
          'bytes', len(serialized), 'sha256', hashlib.sha256(serialized).hexdigest())


if __name__ == '__main__':
    main()
