"""Patched dataset classes for parnet eCLIP datasets in .pt.gz, .pt, and HFDS formats.

Fixes bugs in two parnet library classes:

``parnet.data.datasets.ListDataset`` (fixed by GzListDataset / PreloadedListDataset):

1. ``__init__``: calls ``torch.load(path)`` directly, which fails on gzip-compressed
   files and on PyTorch >= 2.6 (``weights_only=True`` default). ``GzListDataset``
   also auto-detects a companion ``.pt`` file for ``torch.load(mmap=True)``
   loading (~300-500 MB RAM vs ~12-15 GB).
2. ``__getitem__``: calls ``sequence_to_onehot()`` expecting a DNA string, but pt.gz
   stores sequences as dense ``(L, 4)`` tensors, silently producing all-zero encodings.
3. **Center-pad convention**: ``ListDataset`` pads with *floor-left, ceil-right* for
   ``pad_side=0`` with odd total padding, which does not match the pre-padded 600nt
   source data (*ceil-left, floor-right*). The wrapper classes use the correct
   convention so that stripped native-length tiles reconstruct exactly.

``parnet.data.datasets.HFDSDataset`` (fixed by HFDSDataset):

1. Returns a ``(inputs, outputs)`` tuple incompatible with ``LightningModel.training_step``
   (which expects ``batch["inputs"]["sequence"]``).
2. Shuffle is commented out and never applied.
3. ``total_key``/``control_key`` are hardcoded to ``"eCLIP"``/``"control"``.
   ``HFDSDataset`` here makes these parametrizable.

.. note::
   ``LightningModel`` (``parnet.bin.train``) expects the batch to contain
   ``batch["outputs"]["total"]`` and, when ``use_control=True``,
   ``batch["outputs"]["control"]``.  If the parnet library changes this contract,
   the ``__getitem__`` implementations below must be updated accordingly.
   See: parnet/bin/train.py — LightningModel.training_step / validation_step.
"""

from __future__ import annotations

import copy
import gzip
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.data

from parnet.data.datasets import ListDataset, t_sparse_to_dense

_log = logging.getLogger(__name__)


def _load_dataset_metadata(path: Path) -> dict | None:
    """Read the optional metadata.yaml sidecar for a dataset file or directory.

    For .pt / .pt.gz files the sidecar is ``<stem>.metadata.yaml`` in the same
    directory. For HFDS directories the sidecar is ``dataset.metadata.yaml``
    inside the directory. Returns None if no sidecar exists.
    """
    try:
        import yaml
    except ImportError:
        return None

    if path.is_dir():
        candidates = [path / "dataset.metadata.yaml"]
    else:
        stem = path.name
        for suffix in (".pt.gz", ".pt"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidates = [path.parent / f"{stem}.metadata.yaml"]

    for candidate in candidates:
        if candidate.exists():
            try:
                return yaml.safe_load(candidate.read_text())
            except Exception as exc:
                _log.warning("Could not read dataset metadata sidecar %s: %s", candidate, exc)
    return None


def _compute_pad_sizes(
    seq: str,
    pad_side: int | None,
    name: str | bytes,
    target_length: int | None,
) -> tuple[int, int]:
    """Compute ``(pad_left, pad_right)`` for a native-length string sequence.

    Returns ``(0, 0)`` when:

    - ``target_length`` is ``None``
    - ``pad_side`` is ``-1`` or ``None`` (no padding intent)
    - ``len(seq) >= target_length`` (already at or beyond window length)

    Center-pad convention (``pad_side=0``): **ceil-left, floor-right** — matching the
    pre-padded 600nt source data and ``parnet_demo_utils.infer_pad_sizes``.  This differs
    from ``parnet.data.datasets.ListDataset`` (floor-left, ceil-right); for odd total
    padding the difference is one nucleotide, which matters for position-specific
    analyses in a sequence-to-signal model.

    Args:
        seq:           Native-length DNA string.
        pad_side:      ``meta["pad_side"]`` value — 0=center, 1=left, 2=right, -1/None=none.
        name:          Tile name ``"chrN:start-end:strand"`` (bytes accepted).
        target_length: Target window length (e.g. 600).

    Returns:
        ``(pad_left, pad_right)`` — N's to prepend / append.
    """
    if target_length is None or pad_side is None or pad_side == -1:
        return 0, 0
    total = target_length - len(seq)
    if total <= 0:
        return 0, 0
    if pad_side == 1:
        pad_left, pad_right = total, 0
    elif pad_side == 2:
        pad_left, pad_right = 0, total
    else:  # 0 = center — ceil-left, floor-right (matches pre-padded 600nt data)
        pad_right = total // 2
        pad_left  = total - pad_right
    assert pad_left >= 0 and pad_right >= 0
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    if name.endswith("-") and pad_side in (1, 2):
        pad_left, pad_right = pad_right, pad_left
    return pad_left, pad_right


def _getitem_impl(self: Any, idx: int) -> dict[str, Any]:
    """Shared __getitem__ implementation for GzListDataset and PreloadedListDataset.

    Handles two sequence storage formats:

    - **Dense tensor** ``(L, 4)`` (old .pt.gz): converted directly, no padding applied
      (assumed pre-padded to window length).
    - **DNA string**: padding is applied when ``self.length`` is set and
      ``len(seq) < self.length`` — using the correct ceil-left, floor-right convention
      for center padding (``pad_side=0``).

    Args:
        idx: Sample index (ignored when ``self.shuffle`` is True).

    Returns:
        Dict with keys:
            ``inputs.sequence``: float tensor shape ``(4, L)``, channels-first one-hot.
            ``outputs.total``: long tensor — the eCLIP/signal track.
            ``outputs.control``: long tensor — control track (only when
                ``self.control_key`` is not None).
            ``meta``: tile metadata dict (only when ``self.return_meta`` is True).
    """
    if self.shuffle:
        idx = int(torch.randint(0, len(self.data), ()))
    sample = copy.deepcopy(self.data[idx])

    seq = sample["inputs"]["sequence"]
    if isinstance(seq, torch.Tensor):
        # Old .pt.gz format: dense (L, 4) one-hot tensor — already at window length.
        pad_left = pad_right = 0
        seq_tensor = seq.float().T   # (L, 4) channels-last → (4, L) channels-first
    else:
        # DNA string — compute padding from meta if needed.
        meta = sample.get("meta", {})
        pad_left, pad_right = _compute_pad_sizes(
            seq,
            meta.get("pad_side"),
            meta.get("name", ""),
            getattr(self, "length", None),
        )
        if pad_left or pad_right:
            seq = "N" * pad_left + seq + "N" * pad_right
        from parnet.utils import sequence_to_onehot
        seq_tensor = sequence_to_onehot(seq).float()   # (4, L)

    total = t_sparse_to_dense(**sample["outputs"][self.total_key])
    if pad_left or pad_right:
        total = F.pad(total, (pad_left, pad_right), mode="constant", value=0)

    target_len = getattr(self, "length", None)
    if target_len:
        assert seq_tensor.shape[1] == target_len, (
            f"sequence length {seq_tensor.shape[1]} != expected {target_len}"
        )
        assert total.shape[1] == target_len, (
            f"signal length {total.shape[1]} != expected {target_len}"
        )

    ret: dict[str, Any] = {
        "inputs":  {"sequence": seq_tensor},
        "outputs": {"total": total.long()},
    }
    if self.control_key is not None:
        ctrl = t_sparse_to_dense(**sample["outputs"][self.control_key])
        if pad_left or pad_right:
            ctrl = F.pad(ctrl, (pad_left, pad_right), mode="constant", value=0)
        ret["outputs"]["control"] = ctrl.long()
    if self.return_meta:
        ret["meta"] = sample["meta"]
    return ret


class GzListDataset(ListDataset):
    """Dataset for ``.pt.gz`` (or plain ``.pt``) eCLIP files from parnet pipelines.

    Accepts either a plain ``.pt`` directly, or a ``.pt.gz`` with a companion
    ``.pt`` alongside it (created once via ``gunzip -k <file>.pt.gz``).

    Args:
        data_pt: Path to the ``.pt.gz`` dataset file (or a plain ``.pt``).
        split: Dataset split to load — ``"train"``, ``"valid"``, or ``"test"``.
        length: Target sequence length (window size, e.g. 600). Required when loading
            a *stripped* dataset (native-length sequences + ``meta["pad_side"]``);
            padding is applied at load time with the correct ceil-left/floor-right
            convention. If ``None`` (default), auto-detected from the ``.metadata.yaml``
            sidecar's ``seq_len`` field when available, otherwise no padding is applied
            (safe for pre-padded datasets where sequences are already at window length).
        mmap: If True (default), use ``torch.load(mmap=True)`` when a plain ``.pt``
            is accessible (~300-500 MB RAM). If False, always deserialise fully into
            RAM (~12-15 GB).
        shuffle: If True, ``__getitem__`` returns a random sample (for training).
        return_meta: If True, the returned dict includes a ``"meta"`` key.
        total_key: Key name for the signal (eCLIP) track in the source file
            (e.g. ``"eCLIP"`` or ``"total"``). Always mapped to ``"total"`` in
            the output dict to match the LightningModel batch contract.
        control_key: Key name for the control track in the source file. Pass
            ``None`` to omit the control track from the output batch — only safe
            when using ``LightningModel(use_control=False)`` with a model that
            does not produce a control output.

    Note:
        If ``parnet.bin.train.LightningModel`` changes its batch interface,
        this class must be updated accordingly.
    """

    def __init__(
        self,
        data_pt: str | Path,
        split: str,
        *,
        length: int | None = None,
        mmap: bool = True,
        shuffle: bool = False,
        return_meta: bool = False,
        total_key: str = "eCLIP",
        control_key: str | None = "control",
    ) -> None:
        torch.utils.data.Dataset.__init__(self)
        self.shuffle = shuffle
        self.return_meta = return_meta
        self.total_key = total_key
        self.control_key = control_key

        _pt = (
            Path(str(data_pt).replace(".pt.gz", ".pt"))
            if str(data_pt).endswith(".pt.gz")
            else Path(data_pt)
        )
        if mmap and _pt.exists():
            print(f"Loading (mmap) {_pt.name} split='{split}'...", end=" ", flush=True)
            self.data = torch.load(_pt, mmap=True, weights_only=False)[split]
        else:
            # no mmap requested, or .pt.gz with no companion .pt — decompress into RAM
            _open = gzip.open if str(data_pt).endswith(".gz") else open
            print(f"Loading (RAM) {Path(data_pt).name} split='{split}'...", end=" ", flush=True)
            with _open(data_pt, "rb") as f:
                self.data = torch.load(f, weights_only=False)[split]
        print(f"loaded {len(self.data)} samples.")

        self.dataset_metadata = _load_dataset_metadata(Path(data_pt))
        if self.dataset_metadata:
            _log.info(
                "Dataset metadata: seq_format=%s, signal_format=%s, n_tracks=%s, splits=%s",
                self.dataset_metadata.get("sequence_format"),
                self.dataset_metadata.get("signal_format"),
                self.dataset_metadata.get("n_tracks"),
                self.dataset_metadata.get("splits"),
            )
        # Auto-detect window length from sidecar when not provided explicitly.
        self.length: int | None = length or (
            self.dataset_metadata.get("seq_len") if self.dataset_metadata else None
        )

    __getitem__ = _getitem_impl


class HFDSDataset(torch.utils.data.Dataset):
    """Dataset backed by a HuggingFace Arrow DatasetDict (HFDS).

    Fixes three bugs in ``parnet.data.datasets.HFDSDataset``:

    1. **Tuple return** — the parnet class returns ``(inputs, outputs)``; this class
       returns a single dict matching the ``LightningModel`` batch contract.
    2. **Broken shuffle** — parnet's shuffle is commented out; this class uses the same
       random-index approach as ``GzListDataset``.
    3. **Hardcoded keys** — ``total_key`` / ``control_key`` are parametrizable here.

    Handles two on-disk sequence formats automatically:

    - **Sparse one-hot** (old ``parnet_data_v1_full`` HFDS): sequence stored as
      ``{indices: [[pos…],[nuc…]], values: [1…], size: [L, 4]}`` — reconstructed
      via ``torch.sparse_coo_tensor(**seq).to_dense().T``.
    - **DNA string** (new HFDS written by the prepare / convert notebooks): sequence
      stored as a plain string — reconstructed via ``sequence_to_onehot``.

    Args:
        hfds_path: Path to the HuggingFace DatasetDict directory (from
            ``save_to_disk``).
        split: Dataset split to load — ``"train"``, ``"valid"``, or ``"test"``.
        length: Target sequence length (window size, e.g. 600). Required for stripped
            datasets; auto-detected from the HFDS sidecar ``seq_len`` when ``None``.
        shuffle: If True, ``__getitem__`` returns a random sample (for training).
        return_meta: If True, the returned dict includes a ``"meta"`` key with
            ``name`` decoded from bytes to str if necessary.
        total_key: Key name for the signal (eCLIP) track in the HFDS file.
            Always mapped to ``"total"`` in the output dict.
        control_key: Key name for the control track, or ``None`` to omit it.

    Note:
        Requires the ``datasets`` (HuggingFace) package.
        If ``parnet.bin.train.LightningModel`` changes its batch interface,
        ``__getitem__`` in this class must be updated accordingly.
    """

    def __init__(
        self,
        hfds_path: str | Path,
        split: str,
        *,
        length: int | None = None,
        shuffle: bool = False,
        return_meta: bool = False,
        total_key: str = "eCLIP",
        control_key: str | None = "control",
    ) -> None:
        from datasets import load_from_disk  # type: ignore[import]

        torch.utils.data.Dataset.__init__(self)
        print(f"Loading (HFDS) {Path(hfds_path).name} split='{split}'...", end=" ", flush=True)
        self.data = load_from_disk(str(hfds_path))[split]
        self.shuffle = shuffle
        self.return_meta = return_meta
        self.total_key = total_key
        self.control_key = control_key
        print(f"loaded {len(self.data)} samples.")

        self.dataset_metadata = _load_dataset_metadata(Path(hfds_path))
        if self.dataset_metadata:
            _log.info(
                "Dataset metadata: seq_format=%s, signal_format=%s, n_tracks=%s, splits=%s",
                self.dataset_metadata.get("sequence_format"),
                self.dataset_metadata.get("signal_format"),
                self.dataset_metadata.get("n_tracks"),
                self.dataset_metadata.get("splits"),
            )
        self.length: int | None = length or (
            self.dataset_metadata.get("seq_len") if self.dataset_metadata else None
        )

    def __len__(self) -> int:
        return len(self.data)


def _hfds_getitem_impl(self: Any, idx: int) -> dict[str, Any]:
    """Shared __getitem__ for HFDSDataset.

    Handles old sparse one-hot and new DNA string sequence formats.  When
    ``self.length`` is set and the sequence is shorter than the window, padding is
    applied using the ceil-left/floor-right convention (matching the 600nt source
    data, see :func:`_compute_pad_sizes`).

    Args:
        idx: Sample index (ignored when ``self.shuffle`` is True).

    Returns:
        Dict with keys:
            ``inputs.sequence``: float tensor shape ``(4, L)``, channels-first one-hot.
            ``outputs.total``: long tensor — the eCLIP/signal track.
            ``outputs.control``: long tensor — control track (only when
                ``self.control_key`` is not None).
            ``meta``: tile metadata dict (only when ``self.return_meta`` is True).
    """
    if self.shuffle:
        idx = int(torch.randint(0, len(self.data), ()))
    example = self.data[idx]

    # Sequence: auto-detect storage format
    seq_raw = example["inputs"]["sequence"]
    pad_left = pad_right = 0
    if isinstance(seq_raw, dict):
        # Old sparse one-hot format (parnet_data_v1_full HFDS):
        # indices [[pos…],[nuc…]], values [1…1], size [L, 4]
        seq_tensor = (
            torch.sparse_coo_tensor(**seq_raw).to_dense().float().T  # (L, 4) → (4, L)
        )
    else:
        # New format: plain DNA string (or bytes).
        from parnet.utils import sequence_to_onehot
        s = seq_raw.decode("utf-8") if isinstance(seq_raw, bytes) else seq_raw
        meta = dict(example.get("meta", {}))
        if isinstance(meta.get("name"), bytes):
            meta["name"] = meta["name"].decode("utf-8")
        pad_left, pad_right = _compute_pad_sizes(
            s,
            meta.get("pad_side"),
            meta.get("name", ""),
            getattr(self, "length", None),
        )
        if pad_left or pad_right:
            s = "N" * pad_left + s + "N" * pad_right
        seq_tensor = sequence_to_onehot(s).float()   # (4, L)

    # Signal / control outputs are always sparse dicts in both HFDS formats
    total = torch.sparse_coo_tensor(**example["outputs"][self.total_key]).to_dense().long()
    if pad_left or pad_right:
        total = F.pad(total, (pad_left, pad_right), mode="constant", value=0)

    target_len = getattr(self, "length", None)
    if target_len:
        assert seq_tensor.shape[1] == target_len, (
            f"sequence length {seq_tensor.shape[1]} != expected {target_len}"
        )
        assert total.shape[1] == target_len, (
            f"signal length {total.shape[1]} != expected {target_len}"
        )

    ret: dict[str, Any] = {
        "inputs":  {"sequence": seq_tensor},
        "outputs": {"total": total},
    }
    if self.control_key is not None:
        ctrl = torch.sparse_coo_tensor(
            **example["outputs"][self.control_key]
        ).to_dense().long()
        if pad_left or pad_right:
            ctrl = F.pad(ctrl, (pad_left, pad_right), mode="constant", value=0)
        ret["outputs"]["control"] = ctrl
    if self.return_meta:
        meta = dict(example["meta"])
        if isinstance(meta.get("name"), bytes):
            meta["name"] = meta["name"].decode("utf-8")
        ret["meta"] = meta
    return ret


# Patch forward reference: HFDSDataset.__getitem__ was defined before _hfds_getitem_impl
HFDSDataset.__getitem__ = _hfds_getitem_impl


class PreloadedListDataset(ListDataset):
    """Dataset wrapping an already-loaded list of samples — no file I/O.

    Use when the full dataset has been loaded once (e.g. in the evaluate notebook
    where all splits and metadata are read in a single ``torch.load`` call).
    ``__getitem__`` is identical to ``GzListDataset``.

    Args:
        data_list: Pre-loaded list of sample dicts (one split's worth).
        length: Target sequence length (window size, e.g. 600). Required when the
            loaded data is a stripped dataset (native-length sequences). Unlike
            ``GzListDataset``, no sidecar is read — pass the window length explicitly.
        shuffle: If True, ignores the requested index and returns a random sample.
        return_meta: If True, the returned dict includes a ``"meta"`` key.
        total_key: Key name for the signal track in the source dicts
            (mapped to ``"total"`` in the output dict).
        control_key: Key name for the control track, or ``None`` to omit it.

    Note:
        If ``parnet.bin.train.LightningModel`` changes its batch interface,
        this class must be updated accordingly.
    """

    def __init__(
        self,
        data_list: list[dict],
        *,
        length: int | None = None,
        shuffle: bool = False,
        return_meta: bool = False,
        total_key: str = "eCLIP",
        control_key: str | None = "control",
    ) -> None:
        torch.utils.data.Dataset.__init__(self)
        self.data = data_list
        self.length = length
        self.shuffle = shuffle
        self.return_meta = return_meta
        self.total_key = total_key
        self.control_key = control_key

    __getitem__ = _getitem_impl
