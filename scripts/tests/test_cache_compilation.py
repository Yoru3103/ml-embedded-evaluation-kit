# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Protect shared calibrated graphs during cache-capacity experiments."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from model_export.tools.compile_fastcache import CompilationContext, compile_case


class CacheCompilationTests(unittest.TestCase):
    """Check compiler mutations and persisted completion records across capacities."""

    def test_mutating_compiler_does_not_modify_reused_graph(self):
        """Run two cache cases against a compiler that mutates its graph input."""
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source/pte_Dedicated_Sram/best_graphs/quantized.pt2"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source archive fixture")
            config = directory / "vela.ini"
            config.write_text("[fixture]\n")
            options = argparse.Namespace(vela_config=config, target="ethos-u85-1024",
                                         system_config="fixture")
            args = argparse.Namespace(output_root=directory / "compiled",
                                      model_root=directory / "source")
            graph = torch.fx.symbolic_trace(torch.nn.ReLU())
            graph.meta = {}
            context = CompilationContext(options, [], {"image_size": 32, "num_classes": 1},
                                         graph)
            observed = []

            def mutate(candidate: torch.fx.GraphModule, samples: list[dict],
                       metadata: dict, settings: argparse.Namespace) -> dict:
                """Simulate a backend consuming and modifying its input graph.

                :param candidate:   Per-capacity graph.
                :param samples:     Shared reference samples.
                :param metadata:    Shared output contract.
                :param settings:    Destination and compilation settings.
                :returns:           Compiled artifact location.
                """
                self.assertEqual(samples, context.samples)
                self.assertEqual(metadata, context.metadata)
                self.assertNotIn("compiled", candidate.meta)
                candidate.meta["compiled"] = True
                observed.append(candidate)
                settings.output.write_bytes(b"compiled fixture")
                return {"path": str(settings.output)}

            with patch("model_export.tools.compile_fastcache.compile_quantized_pte",
                       side_effect=mutate):
                for capacity in (64, 128):
                    compile_case("pte", capacity, args, context)
                    record = json.loads((args.output_root / "pte" / f"{capacity}k"
                                         / "compilation.json").read_text())
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual(record["cache_bytes"], capacity * 1024)
            self.assertEqual(graph.meta, {})
            self.assertIsNot(observed[0], observed[1])


if __name__ == "__main__":
    unittest.main()
