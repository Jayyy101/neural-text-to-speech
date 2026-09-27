"""Fixed Mandarin ASR completeness gate and audio-only worker client."""

import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata


MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
MODEL_REVISION = "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd"
QC_POLICY = "mandarin_asr_contiguous_han_deletion_v1"
DECODER_POLICY = "greedy_ctc_argmax_collapse_repeats_remove_blank_preserve_unk_v1"
COMPARISON_POLICY = "unicode_nfc_han_only_preserve_unk_order_v1"
ALIGNMENT_POLICY = "unit_cost_levenshtein_tie_substitution_deletion_insertion_v1"
DELETION_THRESHOLD = 4
MAX_WHOLE_WAV_SECONDS = 30.0
DEFAULT_ASR_PYTHON = "/home/jay/miniconda3/envs/tts-align/bin/python"


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def policy_record(asr_python=DEFAULT_ASR_PYTHON):
    return {
        "policy": QC_POLICY, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "decoder_policy": DECODER_POLICY,
        "comparison_policy": COMPARISON_POLICY,
        "alignment_policy": ALIGNMENT_POLICY,
        "contiguous_expected_han_deletion_threshold": DELETION_THRESHOLD,
        "max_whole_wav_seconds": MAX_WHOLE_WAV_SECONDS,
        "asr_python": str(asr_python),
    }


def is_han(character):
    name = unicodedata.name(character, "")
    return (name.startswith("CJK UNIFIED IDEOGRAPH-")
            or name.startswith("CJK COMPATIBILITY IDEOGRAPH-")
            or character == "〇")


def han_tokens(text):
    return [character for character in unicodedata.normalize("NFC", text)
            if is_han(character)]


def levenshtein_steps(expected, recognized):
    """Validated unit-cost alignment with fixed substitution/deletion/insertion ties."""
    n, m = len(expected), len(recognized)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    choices = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        choices[i][0] = "deletion"
    for j in range(1, m + 1):
        costs[0][j] = j
        choices[0][j] = "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == recognized[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                choices[i][j] = "match"
            else:
                candidates = (
                    (costs[i - 1][j - 1] + 1, 0, "substitution"),
                    (costs[i - 1][j] + 1, 1, "deletion"),
                    (costs[i][j - 1] + 1, 2, "insertion"),
                )
                _, _, operation = min(candidates)
                costs[i][j] = min(item[0] for item in candidates)
                choices[i][j] = operation
    steps = []
    i, j = n, m
    while i or j:
        operation = choices[i][j]
        if operation in ("match", "substitution"):
            steps.append({"operation": operation, "expected_index": i - 1,
                          "recognized_index": j - 1, "expected": expected[i - 1],
                          "recognized": recognized[j - 1]})
            i -= 1
            j -= 1
        elif operation == "deletion":
            steps.append({"operation": operation, "expected_index": i - 1,
                          "recognized_index": None, "recognized_position": j,
                          "expected": expected[i - 1], "recognized": None})
            i -= 1
        elif operation == "insertion":
            steps.append({"operation": operation, "expected_index": None,
                          "expected_position": i, "recognized_index": j - 1,
                          "expected": None, "recognized": recognized[j - 1]})
            j -= 1
        else:
            raise ValueError("Levenshtein traceback is incomplete")
    steps.reverse()
    return steps, costs[n][m]


def _token_time(tokens, index):
    if index is None or not 0 <= index < len(tokens):
        return None
    return [tokens[index]["start_seconds"], tokens[index]["end_seconds"]]


def group_edits(steps, expected, recognized_meta):
    """Group contiguous edit operations exactly as in the validated audit."""
    groups = []
    current = None
    recognized = [item["comparison_token"] for item in recognized_meta]
    for step in steps:
        if step["operation"] == "match":
            if current:
                groups.append(current)
                current = None
            continue
        if current is None or current["operation"] != step["operation"]:
            if current:
                groups.append(current)
            current = {"operation": step["operation"], "steps": []}
        current["steps"].append(step)
    if current:
        groups.append(current)
    records = []
    for index, group in enumerate(groups, 1):
        expected_indices = [step["expected_index"] for step in group["steps"]
                            if step["expected_index"] is not None]
        recognized_indices = [step["recognized_index"] for step in group["steps"]
                              if step["recognized_index"] is not None]
        if expected_indices:
            expected_start = min(expected_indices)
            expected_end = max(expected_indices) + 1
        else:
            expected_start = expected_end = group["steps"][0]["expected_position"]
        if recognized_indices:
            recognized_start = min(recognized_indices)
            recognized_end = max(recognized_indices) + 1
        else:
            recognized_start = recognized_end = group["steps"][0]["recognized_position"]
        records.append({
            "group_index": index, "operation": group["operation"],
            "expected_start": expected_start, "expected_end": expected_end,
            "expected_text": "".join(expected[i] for i in expected_indices),
            "recognized_start": recognized_start, "recognized_end": recognized_end,
            "recognized_text": "".join(recognized[i] for i in recognized_indices),
            "expected_context": "".join(expected[max(0, expected_start - 8):
                                               min(len(expected), expected_end + 8)]),
            "recognized_context": "".join(recognized[max(0, recognized_start - 8):
                                                   min(len(recognized), recognized_end + 8)]),
            "recognized_time_span_seconds": (
                [recognized_meta[recognized_start]["start_seconds"],
                 recognized_meta[recognized_end - 1]["end_seconds"]]
                if recognized_indices else None
            ),
            "deletion_neighbor_times_seconds": (
                {"left": _token_time(recognized_meta, recognized_start - 1),
                 "right": _token_time(recognized_meta, recognized_start)}
                if group["operation"] == "deletion" else None
            ),
        })
    return records


def compare_recognition(normalized_text, recognition):
    expected = han_tokens(normalized_text)
    if not expected:
        raise ValueError("Frozen unit has no expected Han comparison characters.")
    recognized_meta = recognition["comparison_tokens"]
    recognized = [item["comparison_token"] for item in recognized_meta]
    if any(token != "<unk>" and (len(token) != 1 or not is_han(token))
           for token in recognized):
        raise ValueError("ASR comparison tokens contain an unsupported value.")
    steps, distance = levenshtein_steps(expected, recognized)
    counts = {operation: sum(step["operation"] == operation for step in steps)
              for operation in ("match", "deletion", "insertion", "substitution")}
    groups = group_edits(steps, expected, recognized_meta)
    flagged = [group for group in groups
               if group["operation"] == "deletion"
               and group["expected_end"] - group["expected_start"] >= DELETION_THRESHOLD]
    recognized_han = [token for token in recognized if token != "<unk>"]
    return {
        "expected_han_sha256": sha256_text("".join(expected)),
        "expected_han_count": len(expected),
        "recognized_han_sha256": sha256_text("".join(recognized_han)),
        "recognized_han_count": len(recognized_han),
        "recognized_comparison_sha256": sha256_text("".join(recognized)),
        "recognized_comparison_count": len(recognized),
        "recognized_comparison_text": "".join(recognized),
        "edit_distance": distance,
        "cer": distance / len(expected),
        "counts": counts,
        "edit_groups": groups,
        "flagged_deletion_groups": flagged,
        "decision": "rejected" if flagged else "passed",
    }


def audio_request(path, wav_sha256, request_id):
    """Worker protocol intentionally has no text-bearing fields."""
    return {"type": "recognize", "request_id": request_id,
            "audio_path": str(Path(path).resolve()), "wav_sha256": wav_sha256}


class ASRWorkerClient:
    """One serial, long-lived ASR subprocess in the isolated tts-align env."""

    def __init__(self, asr_python, log_path):
        self.asr_python = str(asr_python)
        self.log_path = Path(log_path)
        self.process = None
        self.log = None
        self.model = None

    def start(self):
        if self.process is not None:
            return self.model
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [self.asr_python, "-B", "-m", "src.audiobook.asr_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1,
        )
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ASR worker exited before its ready message.")
        ready = json.loads(line)
        if (ready.get("type") != "ready"
                or ready.get("model", {}).get("model_id") != MODEL_ID
                or ready.get("model", {}).get("resolved_revision") != MODEL_REVISION):
            raise RuntimeError("ASR worker model revision or startup contract differs.")
        self.model = ready["model"]
        return self.model

    def recognize(self, request):
        if set(request) != {"type", "request_id", "audio_path", "wav_sha256"}:
            raise ValueError("ASR request contains unsupported fields.")
        self.start()
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ASR worker exited during recognition.")
        response = json.loads(line)
        if response.get("request_id") != request["request_id"]:
            raise RuntimeError("ASR worker response identity differs.")
        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "ASR worker failed."))
        if response.get("type") != "recognized" or response.get("wav_sha256") != request["wav_sha256"]:
            raise RuntimeError("ASR worker response audio identity differs.")
        return response

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
            finally:
                self.process.stdout.close()
                self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
