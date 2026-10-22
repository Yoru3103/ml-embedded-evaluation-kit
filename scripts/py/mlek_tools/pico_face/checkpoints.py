# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Pinned checkpoint identities; hashes are verified before loading."""
import hashlib
from pathlib import Path

PICO_FACES_REPO = "cpldcpu/pico-faces"
PICO_FACES_COMMIT = "ee15d9d183d8428efaeb3078edf47669cfc23994"  # 2026-09-12, adds the LICENSE
PICO_FACES_DIR = Path(__file__).parent / "pico_faces"
PICO_FACES_FILES: dict[str, dict[str, str]] = {
    "m3_long_cfg": {
        "dit_qat.pt": "d645192896db72e71905d65d465e9c08384502336f9d02811d476743efac651a",
        "vae_final.pt": "273d0d1b1784dec8d122642fd6dc028fc1427763f4b0fcbfdb86853621e46646",
        "latent_stats.npz": "3cb07560a18a3e7a4b448e99c4ff00cf531cf51fe8d0ce2e97ab00f2b4113470",
    },
    "m3_decD_deep_full": {
        "dit_qat.pt": "430ca1ef03c52008e46e477ecdbbcf47d61b8588ebee1556e955bfe52c1905e3",
        "vae_final.pt": "3dcd8e09a8717f5011373c77a64e18bd83cfa493a7a6cda6a79d56d7034ef3db",
        "latent_stats.npz": "3cb07560a18a3e7a4b448e99c4ff00cf531cf51fe8d0ce2e97ab00f2b4113470",
    },
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
