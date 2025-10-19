# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from typing import Any, TypeVar

import numpy as np
import torch
from copy import deepcopy

from monai import config, transforms
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms.traits import LazyTrait, RandomizableTrait, ThreadUnsafe, MultiSampleTrait
from monai.utils import MAX_SEED, ensure_tuple, first
from monai.utils.enums import TransformBackends
from monai.utils.misc import MONAIEnvVars
from monai.transforms.transform import Randomizable, MapTransform, LazyTransform
from monai.transforms import RandCropByPosNegLabel

class RandCropByPosNegLabeld_Custom(Randomizable, MapTransform, LazyTransform, MultiSampleTrait):
    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Sequence[int] | int,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        fg_indices_key: str | None = None,
        bg_indices_key: str | None = None,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.image_key = image_key
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.cropper = RandCropByPosNegLabel(
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCropByPosNegLabeld:
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(
        self,
        label: torch.Tensor | None = None,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        image: torch.Tensor | None = None,
    ) -> None:
        # bg_indices가 None인 경우 전체 영역에서 bg 후보를 생성
        if bg_indices is None:
            if image is not None:
                # image가 주어지면, image의 첫 채널 또는 합산 결과에서 image_threshold를 넘는 영역을 사용
                if image.shape[0] == 1:
                    img_np = image[0].cpu().numpy()
                else:
                    img_np = image.sum(dim=0).cpu().numpy()
                valid_mask = img_np > self.cropper.image_threshold
                bg_indices = np.where(valid_mask.flatten())[0]
            else:
                # image가 없으면 label의 전체 영역(모든 voxel)을 negative 후보로 사용
                if label is not None:
                    shape = label.shape[1:]
                    bg_indices = np.arange(np.prod(shape))
                else:
                    raise ValueError("Either image or label must be provided for bg_indices.")
        self.cropper.randomize(label=label, fg_indices=fg_indices, bg_indices=bg_indices, image=image)

    @LazyTransform.lazy_evaluation.setter  # type: ignore
    def lazy_evaluation(self, value: bool) -> None:
        self._lazy_evaluation = value
        self.cropper.lazy_evaluation = value

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        fg_indices = d.pop(self.fg_indices_key, None)
        bg_indices = d.pop(self.bg_indices_key, None)

        self.randomize(d.get(self.label_key), fg_indices, bg_indices, d.get(self.image_key))

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        for key in self.key_iterator(d):
            for i, im in enumerate(self.cropper(d[key], randomize=False)):
                ret[i][key] = im
        return ret
