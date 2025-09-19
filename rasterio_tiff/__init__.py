"""
Rasterio TIFF Editor

メモリ効率の良い巨大TIFFファイルの部分編集を可能にするライブラリ。
"""

from .tiffeditor import TiffEditor, Range, Rect, create_sample_tiff

__version__ = "0.1.0"
__author__ = "User"
__email__ = ""

__all__ = [
    "TiffEditor",
    "Range", 
    "Rect",
    "create_sample_tiff",
]
