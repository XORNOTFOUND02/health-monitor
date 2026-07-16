"""
Session-level train/validation/test splitter.

Splits session IDs into train, validation, and test sets while ensuring
no data leakage -- all windows from a single session stay in the same split.
This is critical because overlapping windows from the same session are
highly correlated and would cause optimistic bias if split across sets.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SessionSplitter:
    """Split session IDs into train/validation/test sets.

    All splits operate at the *session* level: every window belonging to
    a given session is assigned to the same split.  This prevents data
    leakage from the overlapping sliding-window approach.

    Three splitting strategies are provided:

    * :meth:`split_by_ratio` -- simple random shuffle with configurable
      proportions.
    * :meth:`split_by_subject` -- cross-subject evaluation where specified
      subject IDs are placed entirely in the test (and optionally validation)
      set.
    * :meth:`stratified_split` -- stratified split that preserves the
      proportion of each clinical condition across splits.

    Examples
    --------
    >>> splitter = SessionSplitter()
    >>> train_ids, val_ids, test_ids = splitter.split_by_ratio(
    ...     sessions=["s001", "s002", "s003", "s004", "s005"],
    ...     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    ... )
    """

    # ------------------------------------------------------------------
    # Ratio-based split
    # ------------------------------------------------------------------

    def split_by_ratio(
        self,
        sessions: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Randomly shuffle and split sessions by ratio.

        Parameters
        ----------
        sessions : list of str
            Session IDs to split.
        train_ratio : float
            Fraction of sessions for training.
        val_ratio : float
            Fraction of sessions for validation.
        test_ratio : float
            Fraction of sessions for testing.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        tuple of (list, list, list)
            ``(train_ids, val_ids, test_ids)``

        Raises
        ------
        ValueError
            If ratios do not sum to 1.0 or if there are too few sessions.
        """
        self._validate_ratios(train_ratio, val_ratio, test_ratio)

        if len(sessions) == 0:
            logger.warning("Empty session list -- returning empty splits.")
            return [], [], []

        rng = np.random.default_rng(seed)
        shuffled = list(sessions)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))

        # Ensure we do not exceed total
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)
        n_test = n_total - n_train - n_val

        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]

        logger.info(
            "Ratio split: train=%d, val=%d, test=%d (total=%d)",
            len(train_ids),
            len(val_ids),
            len(test_ids),
            n_total,
        )
        return train_ids, val_ids, test_ids

    # ------------------------------------------------------------------
    # Subject-level split
    # ------------------------------------------------------------------

    def split_by_subject(
        self,
        sessions: List[str],
        test_subject_ids: List[str],
        val_subject_ids: Optional[List[str]] = None,
        session_subject_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split sessions by subject ID for cross-subject evaluation.

        All sessions belonging to a test subject are placed in the test set,
        all sessions belonging to a validation subject are placed in the
        validation set, and all remaining sessions go to the training set.

        Parameters
        ----------
        sessions : list of str
            Session IDs to split.
        test_subject_ids : list of str
            Subject IDs whose sessions go to the test set.
        val_subject_ids : list of str or None
            Subject IDs whose sessions go to the validation set.
            ``None`` means no explicit validation subjects.
        session_subject_map : dict or None
            Mapping from session ID to subject ID.  If ``None``, session IDs
            are assumed to *be* subject IDs (each session is a unique subject).
        seed : int
            Random seed used to shuffle remaining training sessions.

        Returns
        -------
        tuple of (list, list, list)
            ``(train_ids, val_ids, test_ids)``
        """
        test_set = set(test_subject_ids)
        val_set = set(val_subject_ids or [])

        # Build mapping: subject -> sessions
        if session_subject_map is not None:
            subject_sessions: Dict[str, List[str]] = defaultdict(list)
            for sid in sessions:
                subj = session_subject_map.get(sid, sid)
                subject_sessions[subj].append(sid)
        else:
            # Each session is its own subject
            subject_sessions = {sid: [sid] for sid in sessions}

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        for subj, subj_sessions in subject_sessions.items():
            if subj in test_set:
                test_ids.extend(subj_sessions)
            elif subj in val_set:
                val_ids.extend(subj_sessions)
            else:
                train_ids.extend(subj_sessions)

        # Shuffle each split for good measure
        rng = np.random.default_rng(seed)
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        logger.info(
            "Subject split: train=%d, val=%d, test=%d "
            "(test_subjects=%s, val_subjects=%s)",
            len(train_ids),
            len(val_ids),
            len(test_ids),
            sorted(test_set),
            sorted(val_set),
        )
        return train_ids, val_ids, test_ids

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------

    def stratified_split(
        self,
        sessions: List[str],
        labels: Dict[str, Dict[str, bool]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Stratified split preserving condition proportions.

        Each session is assigned a *dominant condition label* based on
        whether any of its windows is positive for that condition.  The
        split then ensures each condition is proportionally represented
        across train, validation, and test sets.

        Parameters
        ----------
        sessions : list of str
            Session IDs to split.
        labels : dict
            ``{session_id: {condition_name: bool, ...}}`` per session.
            Typically the session-level aggregation of window labels.
        train_ratio : float
            Fraction for training.
        val_ratio : float
            Fraction for validation.
        test_ratio : float
            Fraction for testing.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        tuple of (list, list, list)
            ``(train_ids, val_ids, test_ids)``

        Raises
        ------
        ValueError
            If ratios do not sum to 1.0.
        """
        self._validate_ratios(train_ratio, val_ratio, test_ratio)

        if len(sessions) == 0:
            return [], [], []

        # Determine a stratification key for each session.
        # Use the first positive condition as the stratum; sessions with
        # no positive condition get a "normal" stratum.
        session_strata: Dict[str, str] = {}
        for sid in sessions:
            sess_labels = labels.get(sid, {})
            positive = [c for c, v in sess_labels.items() if v]
            if positive:
                # Sort for determinism; take the first alphabetically
                session_strata[sid] = sorted(positive)[0]
            else:
                session_strata[sid] = "normal"

        # Group sessions by stratum
        strata_groups: Dict[str, List[str]] = defaultdict(list)
        for sid in sessions:
            strata_groups[session_strata[sid]].append(sid)

        rng = np.random.default_rng(seed)
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        for stratum, group in strata_groups.items():
            rng.shuffle(group)
            n = len(group)
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # Clamp
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)

            train_ids.extend(group[:n_train])
            val_ids.extend(group[n_train : n_train + n_val])
            test_ids.extend(group[n_train + n_val :])

            logger.debug(
                "Stratum '%s': total=%d -> train=%d, val=%d, test=%d",
                stratum,
                n,
                n_train,
                n_val,
                n - n_train - n_val,
            )

        # Shuffle final splits
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        logger.info(
            "Stratified split: train=%d, val=%d, test=%d (strata=%s)",
            len(train_ids),
            len(val_ids),
            len(test_ids),
            sorted(strata_groups.keys()),
        )
        return train_ids, val_ids, test_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ratios(
        train_ratio: float, val_ratio: float, test_ratio: float
    ) -> None:
        """Raise ``ValueError`` if ratios do not sum to approximately 1.0."""
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.6f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

    @staticmethod
    def split_stats(
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        labels: Optional[Dict[str, Dict[str, bool]]] = None,
    ) -> str:
        """Return a human-readable summary of the split.

        Parameters
        ----------
        train_ids, val_ids, test_ids : list of str
            Session IDs for each split.
        labels : dict or None
            Per-session labels.  If provided, per-condition counts are shown.

        Returns
        -------
        str
            Multi-line summary string.
        """
        total = len(train_ids) + len(val_ids) + len(test_ids)
        lines = [
            "Session Split Summary",
            "=" * 40,
            f"  Train : {len(train_ids):5d} sessions ({len(train_ids)/max(total,1)*100:.1f}%)",
            f"  Val   : {len(val_ids):5d} sessions ({len(val_ids)/max(total,1)*100:.1f}%)",
            f"  Test  : {len(test_ids):5d} sessions ({len(test_ids)/max(total,1)*100:.1f}%)",
            f"  Total : {total:5d} sessions",
        ]

        if labels is not None:
            lines.append("")
            lines.append("Per-condition distribution:")
            for split_name, ids in [
                ("Train", train_ids),
                ("Val", val_ids),
                ("Test", test_ids),
            ]:
                cond_counts: Counter[str] = Counter()
                for sid in ids:
                    for cond, present in labels.get(sid, {}).items():
                        if present:
                            cond_counts[cond] += 1
                if cond_counts:
                    items = ", ".join(
                        f"{c}={n}" for c, n in cond_counts.most_common()
                    )
                else:
                    items = "(no conditions)"
                lines.append(f"  {split_name:5s}: {items}")

        return "\n".join(lines)
