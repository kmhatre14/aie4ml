# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Final, Union


class RoundingMode(Enum):
    TRN = 'TRN'
    RND_MIN_INF = 'RND_MIN_INF'
    TRN_ZERO = 'TRN_ZERO'
    RND_ZERO = 'RND_ZERO'
    RND_INF = 'RND_INF'
    RND_CONV = 'RND_CONV'
    RND = 'RND'


class SaturationMode(Enum):
    WRAP = 'WRAP'
    SAT = 'SAT'
    SAT_ZERO = 'SAT_ZERO'
    SAT_SYM = 'SAT_SYM'


@dataclass(frozen=True)
class QuantIntent:
    width: int
    frac: int
    signed: bool
    rounding: RoundingMode
    saturation: SaturationMode


class FloatFormat(Enum):
    BF16 = 'bfloat16'
    FP32 = 'float32'
    FP8_E4M3 = 'fp8_e4m3'


_FORMAT_WIDTHS: Final[dict[str, int]] = {
    'int4': 4,
    'uint4': 4,
    'int8': 8,
    'uint8': 8,
    'int16': 16,
    'uint16': 16,
    'int32': 32,
    'uint32': 32,
    'int64': 64,
    'uint64': 64,
    FloatFormat.BF16.value: 16,
    FloatFormat.FP32.value: 32,
    FloatFormat.FP8_E4M3.value: 8,
    'accfloat': 32,
}
_FORMAT_SIGNED: Final[dict[str, bool]] = {
    'int4': True,
    'uint4': False,
    'int8': True,
    'uint8': False,
    'int16': True,
    'uint16': False,
    'int32': True,
    'uint32': False,
    'int64': True,
    'uint64': False,
    FloatFormat.BF16.value: True,
    FloatFormat.FP32.value: True,
    FloatFormat.FP8_E4M3.value: True,
    'accfloat': True,
}
_FORMAT_CTYPES: Final[dict[str, str]] = {
    'int4': 'int8_t',
    'uint4': 'uint8_t',
    'int8': 'int8_t',
    'uint8': 'uint8_t',
    'int16': 'int16_t',
    'uint16': 'uint16_t',
    'int32': 'int32_t',
    'uint32': 'uint32_t',
    'int64': 'int64_t',
    'uint64': 'uint64_t',
    FloatFormat.BF16.value: 'bfloat16',
    FloatFormat.FP32.value: 'float',
    FloatFormat.FP8_E4M3.value: 'float8',
    'accfloat': 'accfloat',
}
FLOAT_FORMATS: Final[frozenset[str]] = frozenset(
    {FloatFormat.BF16.value, FloatFormat.FP32.value, FloatFormat.FP8_E4M3.value}
)
FLOAT_LIKE_FORMATS: Final[frozenset[str]] = frozenset(FLOAT_FORMATS | {'accfloat'})


@dataclass(frozen=True)
class FloatIntent:
    width: int
    format: FloatFormat


PrecisionIntent = Union[QuantIntent, FloatIntent]


def width_for_format(format_str: str) -> int:
    try:
        return int(_FORMAT_WIDTHS[format_str])
    except KeyError as exc:
        raise ValueError(f'Unknown AIEDataType format: {format_str!r}') from exc


def ctype_for_format(format_str: str) -> str:
    try:
        return _FORMAT_CTYPES[format_str]
    except KeyError as exc:
        raise ValueError(f'Unknown AIEDataType format: {format_str!r}') from exc


def legality_format(format_str: str) -> str:
    if format_str.startswith('uint'):
        return f'int{format_str[4:]}'
    return format_str


def signed_for_format(format_str: str) -> bool:
    try:
        return bool(_FORMAT_SIGNED[format_str])
    except KeyError as exc:
        raise ValueError(f'Unknown AIEDataType format: {format_str!r}') from exc


@dataclass(frozen=True)
class AIEDataType:
    format: str
    frac: int = 0
    rounding: RoundingMode = RoundingMode.RND
    saturation: SaturationMode = SaturationMode.SAT

    def __post_init__(self) -> None:
        if self.format in FLOAT_LIKE_FORMATS and int(self.frac) != 0:
            raise ValueError(f'AIEDataType format {self.format!r} requires frac=0.')

    @property
    def width(self) -> int:
        return width_for_format(self.format)

    @property
    def signed(self) -> bool:
        return signed_for_format(self.format)

    @property
    def c_type(self) -> str:
        return ctype_for_format(self.format)

    @property
    def storage_dtype(self) -> str:
        """C type for static array storage (may differ from c_type for packed formats)."""
        if self.format == 'bfloat16':
            return 'uint16_t'
        if self.format == 'fp8_e4m3':
            return 'uint8_t'
        return self.c_type

    def to_dict(self) -> dict:
        return {
            'format': self.format,
            'frac': int(self.frac),
            'rounding': self.rounding.name,
            'saturation': self.saturation.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AIEDataType':
        return cls(
            format=str(data['format']),
            frac=int(data['frac']),
            rounding=RoundingMode[str(data['rounding'])],
            saturation=SaturationMode[str(data['saturation'])],
        )
