"""GLiNER2.5 ``classify_text`` as a local fixed-option decision backend.

:class:`GLiNER2DecisionBackend` plugs a loaded GLiNER2.5 checkpoint into the
decision contract in :mod:`openmed.structured.decision`. Each request becomes
one ``classify_text`` task over the caller's options, and the backend returns
the probability of every option so the contract's calibrated abstention can
judge confidence and margin:

* ``fixed_choice`` and ``ordered_preference``: a single-label task; the
  scores are the softmax distribution over the options.
* ``boolean_choice``: options that read as yes/no (``yes``/``no``,
  ``true``/``false``, ``y``/``n``) are asked as the checkpoint's yes/no labels
  and mapped back to the caller's wording and order; any other pair is asked
  as a two-label single-label task.
* ``multi_label``: a ``multi_label`` task with independent sigmoid scores for
  every option. The calibration profile's ``multi_label_threshold`` selects
  labels; the backend never drops low-scoring options.
* ``scalar_score``: the probability of the affirmative yes/no label, with the
  larger of the two probabilities as confidence.

Results stay assistive review artifacts: the contract always requires human
review and never authorizes an autonomous clinical action. Clinical uses go
through ``openmed.clinical.decision_support.GuardedSuggestion``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any, Optional

from openmed.structured.decision import (
    DecisionConflictError,
    DecisionError,
    DecisionMode,
    DecisionUnsupportedError,
    EncoderDecisionBackend,
)

from .gliner2 import (
    GLiNER2Classification,
    GLiNER2Handle,
    GLiNER2OutputError,
    load_gliner2_handle,
)

DEFAULT_DECISION_TASK = "decision"
"""Default classification task name, used as the question prompt."""

DEFAULT_BOOLEAN_LABELS = ("yes", "no")
"""Affirmative and negative labels the GLiNER2.5 checkpoints answer."""

GLINER2_DECISION_BACKEND_ID = "openmed.gliner2.classify_text.v1"
"""Default backend identifier recorded in every decision result."""

_AFFIRMATIVE = frozenset({"yes", "y", "true"})
_NEGATIVE = frozenset({"no", "n", "false"})


class GLiNER2DecisionBackend(EncoderDecisionBackend):
    """Encoder decision backend that scores options with GLiNER2.5.

    The backend identity records the checkpoint's model id, resolved
    revision and license. Construction fails closed when the license is
    unknown or not permissive.

    Args:
        handle: A loaded :class:`~openmed.ner.families.gliner2.GLiNER2Handle`.
        task: Classification task name sent with every request. GLiNER2 reads
            it as part of the prompt, so a short question-like name (for
            example ``"needs_follow_up"``) can sharpen boolean decisions.
        boolean_labels: Affirmative and negative labels used for yes/no
            options and scalar scores.
        backend_id: Identifier recorded in the decision result.
        license_id: SPDX license overriding the handle's license.
        model_id: Model id overriding the handle's path-free model id.
        revision: Revision overriding the handle's resolved revision.

    Raises:
        DecisionError: If the license is unknown or not permissive, or the
            task or labels are invalid.
    """

    def __init__(
        self,
        handle: GLiNER2Handle,
        *,
        task: str = DEFAULT_DECISION_TASK,
        boolean_labels: tuple[str, str] = DEFAULT_BOOLEAN_LABELS,
        backend_id: str = GLINER2_DECISION_BACKEND_ID,
        license_id: Optional[str] = None,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        if not isinstance(handle, GLiNER2Handle):
            raise TypeError("handle must be a GLiNER2Handle")
        self.handle = handle
        self.task = _checked_label(task, "decision task")
        if (
            not isinstance(boolean_labels, tuple)
            or len(boolean_labels) != 2
            or _normalized(boolean_labels[0]) == _normalized(boolean_labels[1])
        ):
            raise DecisionError("boolean_labels must be two distinct labels")
        self.boolean_labels = (
            _checked_label(boolean_labels[0], "boolean label"),
            _checked_label(boolean_labels[1], "boolean label"),
        )
        resolved_license = license_id or handle.license_id
        if not resolved_license:
            raise DecisionError(
                "GLiNER2 decision backends need a permissive license identifier"
            )
        super().__init__(
            backend_id=backend_id,
            model_id=model_id or handle.public_model_id,
            revision=revision or handle.revision or "unknown",
            license_id=resolved_license,
            scorer=self._score,
            runtime="gliner2",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        license_id: Optional[str] = None,
        task: str = DEFAULT_DECISION_TASK,
        boolean_labels: tuple[str, str] = DEFAULT_BOOLEAN_LABELS,
        backend_id: str = GLINER2_DECISION_BACKEND_ID,
    ) -> "GLiNER2DecisionBackend":
        """Load a checkpoint with :func:`load_gliner2_handle` and wrap it.

        Args:
            model_id: Hugging Face repository id or local checkpoint directory.
            cache_dir: Optional Hugging Face cache directory.
            token: Optional Hub token.
            device: Optional torch device.
            revision: Optional branch, tag or commit to pin.
            local_files_only: Never download; ``None`` follows
                ``OPENMED_OFFLINE``.
            license_id: SPDX license for checkpoints outside the catalog.
            task: Classification task name sent with every request.
            boolean_labels: Affirmative and negative labels.
            backend_id: Identifier recorded in the decision result.

        Returns:
            A ready :class:`GLiNER2DecisionBackend`.
        """

        handle = load_gliner2_handle(
            model_id,
            cache_dir=cache_dir,
            token=token,
            device=device,
            revision=revision,
            local_files_only=local_files_only,
            license_id=license_id,
        )
        return cls(
            handle,
            task=task,
            boolean_labels=boolean_labels,
            backend_id=backend_id,
        )

    # -- scoring ------------------------------------------------------------

    def _score(
        self,
        input_text: str,
        options: tuple[str, ...],
        mode: DecisionMode,
    ) -> Any:
        try:
            if mode in (DecisionMode.FIXED_CHOICE, DecisionMode.ORDERED_PREFERENCE):
                return self._exclusive_scores(input_text, options)
            if mode is DecisionMode.BOOLEAN_CHOICE:
                return self._boolean_scores(input_text, options)
            if mode is DecisionMode.MULTI_LABEL:
                return self._multi_label_scores(input_text, options)
            if mode is DecisionMode.SCALAR_SCORE:
                return self._scalar_score(input_text)
        except GLiNER2OutputError:
            raise DecisionConflictError(
                "GLiNER2 scores do not match the decision options"
            ) from None
        raise DecisionUnsupportedError("GLiNER2 cannot score this decision mode")

    def _classify(
        self,
        input_text: str,
        labels: tuple[str, ...],
        *,
        multi_label: bool,
    ) -> GLiNER2Classification:
        result = self.handle.classify_text(
            input_text,
            {self.task: {"labels": list(labels), "multi_label": multi_label}},
        )
        classification = result.get(self.task)
        if classification is None or tuple(
            item.label for item in classification.scores
        ) != tuple(labels):
            raise GLiNER2OutputError("GLiNER2 did not score every decision option")
        return classification

    def _exclusive_scores(
        self, input_text: str, options: tuple[str, ...]
    ) -> tuple[float, ...]:
        classification = self._classify(input_text, options, multi_label=False)
        return tuple(item.score for item in classification.scores)

    def _multi_label_scores(
        self, input_text: str, options: tuple[str, ...]
    ) -> tuple[float, ...]:
        classification = self._classify(input_text, options, multi_label=True)
        return tuple(item.score for item in classification.scores)

    def _boolean_scores(
        self, input_text: str, options: tuple[str, ...]
    ) -> tuple[float, ...]:
        polarity = _boolean_polarity(options)
        if polarity is None:
            return self._exclusive_scores(input_text, options)
        affirmative, negative = self._yes_no(input_text)
        return tuple(affirmative if positive else negative for positive in polarity)

    def _scalar_score(self, input_text: str) -> Mapping[str, Any]:
        affirmative, negative = self._yes_no(input_text)
        return {
            "scalar_score": affirmative,
            "confidence": max(affirmative, negative),
        }

    def _yes_no(self, input_text: str) -> tuple[float, float]:
        classification = self._classify(
            input_text, self.boolean_labels, multi_label=False
        )
        return classification.scores[0].score, classification.scores[1].score


def _normalized(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _checked_label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise DecisionError(f"{name} must be bounded non-empty text")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise DecisionError(f"{name} cannot contain control characters")
    return value


def _boolean_polarity(options: tuple[str, ...]) -> Optional[tuple[bool, bool]]:
    """Return which option is affirmative when the pair reads as yes/no."""

    if len(options) != 2:
        return None
    first, second = (_normalized(option) for option in options)
    if first in _AFFIRMATIVE and second in _NEGATIVE:
        return (True, False)
    if first in _NEGATIVE and second in _AFFIRMATIVE:
        return (False, True)
    return None


__all__ = [
    "DEFAULT_BOOLEAN_LABELS",
    "DEFAULT_DECISION_TASK",
    "GLINER2_DECISION_BACKEND_ID",
    "GLiNER2DecisionBackend",
]
