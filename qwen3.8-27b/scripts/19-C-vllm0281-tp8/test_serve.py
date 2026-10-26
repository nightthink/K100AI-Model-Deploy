"""验证19线的同配置边界；伪载体不接触Docker或DCU。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
IMAGE = "harbor.sourcefind.cn:5443/hcu/admin/base/custom@sha256:2eff683dd747db3484a1b523c05e49d75bb45ed6724b4d1238344385c7e88c9d"


class ServeContractTests(unittest.TestCase):
    def launch(self, **overrides):
        with tempfile.TemporaryDirectory(prefix="q38-line19-", dir="/tmp/opencode") as directory:
            root = Path(directory)
            (root / "kit").mkdir()
            (root / "kit" / "VERSION").write_text("q38-kit-19 fixture-package\n")
            docker = root / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "from pathlib import Path\n"
                "args=sys.argv[1:]\n"
                "with Path(os.environ['CAPTURE']).open('a') as output:\n"
                "    output.write(json.dumps(args)+'\\n')\n"
                "if args and args[0]=='info': sys.exit(0)\n"
                "if args and args[0]=='run': print('fake-container'); sys.exit(0)\n"
                "sys.exit(1)\n"
            )
            docker.chmod(0o755)
            capture = root / "commands.jsonl"
            environment = {
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "HOME": str(root), "WORK": str(root / "kit"), "CFGDIR": str(HERE),
                "MODELS_ROOT": "/models-fixture", "NAME": "01ma.cli-model-svc-line19-fixture",
                "LAUNCH_ID": "fixture-launch", "CAPTURE": str(capture), **overrides,
            }
            result = subprocess.run(["bash", str(HERE / "serve.sh")], env=environment,
                                    capture_output=True, text=True, timeout=15)
            calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
            return result, calls

    def run_arguments(self, **overrides):
        result, calls = self.launch(**overrides)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = [call for call in calls if call[0] == "run"]
        self.assertEqual(len(runs), 1)
        self.assertFalse(any(call[0] in ("rm", "stop", "restart") for call in calls))
        return runs[0]

    @staticmethod
    def values(arguments, option):
        return [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option]

    def test_defaults_match_eight_card_bf16_long_context_workload(self):
        arguments = self.run_arguments()
        self.assertIn(IMAGE, arguments)
        for flag, expected in (("--tensor-parallel-size", "8"), ("--dtype", "bfloat16"),
                               ("--kv-cache-dtype", "bfloat16"), ("--max-model-len", "1000000"),
                               ("--max-num-batched-tokens", "16384"), ("--gpu-memory-utilization", "0.90"),
                               ("--host", "127.0.0.1"), ("--port", "8117")):
            self.assertEqual(self.values(arguments, flag), [expected])
        self.assertIn("--enable-chunked-prefill", arguments)
        self.assertIn("HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", self.values(arguments, "-e"))

    def test_farm_and_baseline_tokenizer_mounts_are_readonly(self):
        arguments = self.run_arguments()
        mounts = self.values(arguments, "-v")
        for mount in ("/models-fixture:/models-fixture:ro",
                      "/models-fixture/Qwen3.8-27B-1M:/models/target:ro",
                      "/models-fixture/Qwen3.8-27B-Channel-INT8-w8a8-hygon/tokenizer.json:/models/target/tokenizer.json:ro",
                      "/models-fixture/Qwen3.8-27B-Channel-INT8-w8a8-hygon/tokenizer_config.json:/models/target/tokenizer_config.json:ro"):
            self.assertIn(mount, mounts)
        self.assertIn(str(HERE) + ":/line:ro", mounts)
        self.assertFalse(any("dist-packages/vllm" in value or "dlhook" in value for value in mounts))

    def test_mtp_prefix_and_parsers_remain_enabled(self):
        arguments = self.run_arguments()
        self.assertEqual(json.loads(self.values(arguments, "--speculative-config")[0]),
                         {"method": "mtp", "num_speculative_tokens": 3})
        for flag in ("--enable-prefix-caching", "--enable-mamba-fine-grained-prefix-cache", "--enable-auto-tool-choice"):
            self.assertIn(flag, arguments)
        for flag, expected in (("--mamba-cache-mode", "align"), ("--block-size", "64"),
                               ("--prefix-match-unit", "64"), ("--attention-backend", "TRITON_ATTN"),
                               ("--reasoning-parser", "qwen3"), ("--tool-call-parser", "qwen3_coder")):
            self.assertEqual(self.values(arguments, flag), [expected])

    def test_math_library_and_explicit_override_reach_container(self):
        defaults = self.values(self.run_arguments(), "-e")
        self.assertIn("ROCBLAS_TENSILE_LIBPATH=/opt/dtk/lib/rocblas/library_gpu4", defaults)
        self.assertIn("VLLM_HCU_USE_CUSTOM_OPS=0", defaults)
        arguments = self.run_arguments(ROCBLAS_TENSILE_LIBPATH="/math-fixture/gfx928")
        self.assertIn("ROCBLAS_TENSILE_LIBPATH=/math-fixture/gfx928", self.values(arguments, "-e"))

    def test_port_override_keeps_full_gpu_topology(self):
        arguments = self.run_arguments(PORT="8129")
        self.assertIn("HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", self.values(arguments, "-e"))
        self.assertEqual(self.values(arguments, "--port"), ["8129"])
        self.assertEqual(self.values(arguments, "--tensor-parallel-size"), ["8"])

    def test_invalid_topology_does_not_create_container(self):
        for gpus in ("0,1", "0,1,2,3", "0,1,2,3,4,5,6,6", "7,6,5,4,3,2,1,0"):
            with self.subTest(gpus=gpus):
                result, calls = self.launch(GPUS=gpus)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("八卡", result.stdout + result.stderr)
                self.assertFalse(any(call[0] == "run" for call in calls))

    def test_identity_labels_and_cache_are_isolated(self):
        arguments = self.run_arguments()
        labels = self.values(arguments, "--label")
        for expected in ("line=19", "publisher=01ma", "launch_id=fixture-launch", "package_id=fixture-package",
                         "gpus=0,1,2,3,4,5,6,7", "port=8117"):
            self.assertIn(expected, labels)
        self.assertTrue(any(value.startswith("model_id=") for value in labels))
        cache_mounts = [value for value in self.values(arguments, "-v") if value.endswith(":/cache")]
        self.assertEqual(len(cache_mounts), 1)
        self.assertIn("/kit/cache-01ma.cli-model-svc-line19-fixture:/cache", cache_mounts[0])


class RuntimeVersionTests(unittest.TestCase):
    def load_check(self):
        specification = importlib.util.spec_from_file_location("q38_line19_runtime", HERE / "runtime_check.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def test_vendor_release_suffix_is_preserved(self):
        module = self.load_check()
        versions = {"vllm": "0.28.1+dtk2604.torch2110.vendor", "vllm_hcu": "0.28.1rc1.dev491+das.example"}
        with patch.object(module.metadata, "version", side_effect=lambda name: versions.get(name, "2.11.0")):
            observed = module.runtime_versions()
        self.assertEqual(observed["vllm"], versions["vllm"])
        self.assertEqual(observed["vllm_hcu"], versions["vllm_hcu"])

    def test_other_release_is_not_accepted_as_0281(self):
        module = self.load_check()
        for wrong in ("0.28.0", "0.28.10", "0.29.0", "1!0.28.1"):
            with self.subTest(version=wrong), patch.object(module.metadata, "version", return_value=wrong):
                with self.assertRaisesRegex(RuntimeError, "0.28.1"):
                    module.runtime_versions()


if __name__ == "__main__":
    unittest.main()
