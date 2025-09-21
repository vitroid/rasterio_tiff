import logging
import os
from typing import Optional, Tuple, Union
import numpy as np
import tifffile
import rasterio
from rasterio.windows import Window
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio import warp
import cv2


# tiledimageからまるまる移植。


# Range、Rectクラスを直接定義（循環インポート回避）
from dataclasses import dataclass


@dataclass
class Range:
    """数値の範囲を表現するクラス"""

    min_val: int
    max_val: int

    @property
    def width(self) -> int:
        """範囲の幅を返す"""
        return self.max_val - self.min_val


@dataclass
class Rect:
    """2次元の領域を表現するクラス"""

    x_range: Range
    y_range: Range

    @classmethod
    def from_bounds(cls, left: int, right: int, top: int, bottom: int) -> "Rect":
        """座標からRectを作成する"""
        return cls(Range(left, right), Range(top, bottom))


class TiffEditor:
    """
    TIFFファイルの部分編集を可能にするクラス（BGR形式でデータを扱う）

    メモリ効率の良い方法で巨大なTIFFファイルの部分的な読み書きを行う。
    TiledImageの設計思想を参考に、ディスク上のTIFFファイルを直接操作する。
    OpenCV（cv2）との互換性のため、すべてのカラー画像データはBGR形式で扱う。

    Features:
    - 部分的な読み込み（メモリ効率、BGR形式で返す）
    - 部分的な書き込み（既存ファイルの更新、BGR形式で受け取る）
    - タイル構造の活用
    - スライス記法での操作
    - OpenCV（cv2）との完全互換
    """

    def __init__(
        self,
        filepath: str,
        mode: str = "r+",
        tilesize: Union[int, Tuple[int, int]] = 512,
        dtype: Optional[np.dtype] = None,
        shape: Optional[Tuple[int, int, int]] = None,
        create_if_not_exists: bool = False,
    ):
        """
        TiffEditorを初期化する

        Args:
            filepath: TIFFファイルのパス
            mode: ファイルのオープンモード ('r', 'r+', 'w')
            tilesize: タイルサイズ（int または (width, height)のタプル）
            dtype: データ型（新規作成時）
            shape: 画像の形状 (height, width, channels)（新規作成時）
            create_if_not_exists: ファイルが存在しない場合に新規作成するか
        """
        self.filepath = filepath
        self.mode = mode

        if isinstance(tilesize, int):
            self.tilesize = (tilesize, tilesize)
        else:
            self.tilesize = tilesize

        self._tiff_handle = None
        self._rasterio_handle = None
        self.logger = logging.getLogger(__name__)

        # ファイルが存在しない場合の処理
        if not os.path.exists(filepath):
            if mode == "w" or (create_if_not_exists and mode in ["w", "r+"]):
                if shape is None or dtype is None:
                    raise ValueError("新規作成時はshapeとdtypeを指定してください")
                self._create_tiff_file(shape, dtype)
                # ファイル作成後に再度オープンを試行
                self._open_file()
            elif mode != "w":
                raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        else:
            # 既存ファイルがある場合
            if mode == "w":
                # 書き込みモードの場合は既存ファイルを上書き
                if shape is None or dtype is None:
                    raise ValueError("mode='w'時はshapeとdtypeを指定してください")
                self._create_tiff_file(shape, dtype)
                self._open_file()
            else:
                self._open_file()

    def _create_tiff_file(self, shape: Tuple[int, int, int], dtype: np.dtype):
        """新しいタイル化TIFFファイルを作成する"""
        height, width, channels = shape

        # ダミーデータで初期化
        dummy_data = np.zeros((height, width, channels), dtype=dtype)

        # タイル化TIFFとして保存
        tifffile.imwrite(
            self.filepath,
            dummy_data,
            tile=self.tilesize,
            photometric="rgb" if channels == 3 else "minisblack",
        )

        self.logger.info(f"新しいTIFFファイルを作成しました: {self.filepath}")

    def _open_file(self):
        """ファイルを開く"""
        try:
            if self.mode == "r":
                # 読み込み専用の場合はtifffileを使用
                if os.path.exists(self.filepath):
                    self._tiff_handle = tifffile.TiffFile(self.filepath)
                else:
                    raise FileNotFoundError(
                        f"読み取り用ファイルが存在しません: {self.filepath}"
                    )
            else:
                # 読み書きの場合はrasterioを使用
                if os.path.exists(self.filepath):
                    self._rasterio_handle = rasterio.open(self.filepath, "r+")
                else:
                    raise FileNotFoundError(
                        f"読み書き用ファイルが存在しません: {self.filepath}"
                    )

            # ハンドルが正しく設定されたかチェック
            if not self._tiff_handle and not self._rasterio_handle:
                raise IOError("ファイルハンドルの初期化に失敗しました")

        except Exception as e:
            raise IOError(f"ファイルを開けませんでした: {e}")

    def _ensure_handle_initialized(self):
        """ハンドルが初期化されていることを保証する"""
        if not self._tiff_handle and not self._rasterio_handle:
            self.logger.warning(
                "ハンドルが初期化されていません。再初期化を試行します。"
            )
            self._open_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """ファイルハンドルを閉じる"""
        if self._tiff_handle:
            self._tiff_handle.close()
            self._tiff_handle = None
        if self._rasterio_handle:
            self._rasterio_handle.close()
            self._rasterio_handle = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """画像の形状を取得"""
        # ハンドルが初期化されていることを確認
        self._ensure_handle_initialized()

        if self._rasterio_handle:
            height, width = self._rasterio_handle.height, self._rasterio_handle.width
            channels = self._rasterio_handle.count
        elif self._tiff_handle:
            page = self._tiff_handle.pages[0]
            height, width = page.shape[:2]
            channels = page.shape[2] if len(page.shape) > 2 else 1
        else:
            raise ValueError("ファイルが開かれていません")

        return (height, width, channels)

    @property
    def dtype(self) -> np.dtype:
        """データ型を取得"""
        # ハンドルが初期化されていることを確認
        self._ensure_handle_initialized()

        if self._rasterio_handle:
            return self._rasterio_handle.dtypes[0]
        elif self._tiff_handle:
            return self._tiff_handle.pages[0].dtype
        else:
            raise ValueError("ファイルが開かれていません")

    def _parse_slice(self, key) -> Rect:
        """スライスを解析してRectに変換する（TiledImageと同じ）"""
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("2次元のスライスを指定してください")

        y_slice, x_slice = key
        if not (isinstance(y_slice, slice) and isinstance(x_slice, slice)):
            raise IndexError("スライスを指定してください")

        # 画像の実際のサイズを取得
        height, width, _ = self.shape

        # スライスの開始と終了を取得
        y_start = y_slice.start if y_slice.start is not None else 0
        y_stop = y_slice.stop if y_slice.stop is not None else height
        x_start = x_slice.start if x_slice.start is not None else 0
        x_stop = x_slice.stop if x_slice.stop is not None else width

        # 範囲チェック
        y_start = max(0, min(y_start, height))
        y_stop = max(0, min(y_stop, height))
        x_start = max(0, min(x_start, width))
        x_stop = max(0, min(x_stop, width))

        # ステップは未対応
        if y_slice.step is not None or x_slice.step is not None:
            raise NotImplementedError("ステップ付きスライスには未対応です")

        return Rect.from_bounds(x_start, x_stop, y_start, y_stop)

    def __getitem__(self, key) -> np.ndarray:
        """スライスで領域を取得する（BGR形式で返す）"""
        region = self._parse_slice(key)
        return self.get_region(region)

    def __setitem__(self, key, value: np.ndarray):
        """スライスで領域を設定する（BGR形式で受け取る）"""
        if not isinstance(value, np.ndarray):
            raise TypeError("NumPy配列を指定してください")

        region = self._parse_slice(key)
        self.put_region(region, value)

    def get_region(self, region: Rect) -> np.ndarray:
        """指定された領域のデータを読み込む（BGR形式で返す）"""
        x_start = region.x_range.min_val
        x_stop = region.x_range.max_val
        y_start = region.y_range.min_val
        y_stop = region.y_range.max_val

        width = x_stop - x_start
        height = y_stop - y_start

        if width <= 0 or height <= 0:
            return np.array([])

        if self._rasterio_handle:
            # rasterioを使用した読み込み
            window = Window(x_start, y_start, width, height)
            data = self._rasterio_handle.read(window=window)

            # rasterioは(channels, height, width)で返すので転置
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0]  # 単一チャンネルの場合

        elif self._tiff_handle:
            # tifffileを使用した読み込み
            page = self._tiff_handle.pages[0]
            data = page.asarray()[y_start:y_stop, x_start:x_stop]

        else:
            raise ValueError("ファイルが開かれていません")

        # RGB形式で保存されているデータをBGR形式に変換（CV2互換）
        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        elif data.ndim == 3 and data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)

        self.logger.debug(
            f"領域を読み込みました（BGR形式）: {region}, shape: {data.shape}"
        )
        return data

    def put_region(self, region: Rect, data: np.ndarray):
        """指定された領域にデータを書き込む（入力はBGR形式、内部でRGBに変換）"""
        if self.mode == "r":
            raise ValueError("読み込み専用モードでは書き込みできません")

        x_start = region.x_range.min_val
        x_stop = region.x_range.max_val
        y_start = region.y_range.min_val
        y_stop = region.y_range.max_val

        width = x_stop - x_start
        height = y_stop - y_start

        if width <= 0 or height <= 0:
            return

        # データサイズの検証
        expected_shape = (height, width)
        if data.ndim == 3:
            expected_shape = (height, width, data.shape[2])
        elif data.ndim == 2:
            expected_shape = (height, width)
        else:
            raise ValueError(f"不正なデータ形状: {data.shape}")

        if data.shape[:2] != (height, width):
            raise ValueError(
                f"データサイズが一致しません。期待: {expected_shape}, 実際: {data.shape}"
            )

        # BGR形式で入力されたデータをRGB形式に変換してからTIFFに保存
        write_data = data.copy()
        if data.ndim == 3 and data.shape[2] == 3:
            write_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        elif data.ndim == 3 and data.shape[2] == 4:
            write_data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)

        if self._rasterio_handle:
            # rasterioを使用した書き込み
            window = Window(x_start, y_start, width, height)

            if write_data.ndim == 3:
                # (height, width, channels) -> (channels, height, width)
                formatted_data = np.transpose(write_data, (2, 0, 1))
                for i in range(formatted_data.shape[0]):
                    self._rasterio_handle.write(formatted_data[i], i + 1, window=window)
            else:
                # 単一チャンネル
                self._rasterio_handle.write(write_data, 1, window=window)

        else:
            raise ValueError("書き込みにはrasterioハンドルが必要です")

        self.logger.debug(
            f"領域に書き込みました（BGR→RGB変換済み）: {region}, shape: {data.shape}"
        )

    def get_info(self) -> dict:
        """ファイルの情報を取得する"""
        shape = self.shape
        dtype = self.dtype

        info = {
            "filepath": self.filepath,
            "shape": shape,
            "dtype": str(dtype),
            "size_mb": os.path.getsize(self.filepath) / (1024 * 1024),
            "tilesize": self.tilesize,
        }

        if self._rasterio_handle:
            info.update(
                {
                    "compression": getattr(
                        self._rasterio_handle, "compression", "unknown"
                    ),
                    "photometric": getattr(
                        self._rasterio_handle, "photometric", "unknown"
                    ),
                    "is_tiled": getattr(self._rasterio_handle, "is_tiled", False),
                }
            )

        return info

    def get_scaled_image(
        self,
        scale_factor: Optional[float] = None,
        target_width: Optional[int] = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        """
        TIFF画像全体の縮小版を効率的に取得する（読み取り専用対応）

        Args:
            scale_factor: 縮小倍率（0.0-1.0）。target_widthが指定された場合は無視される
            target_width: 目標となる画像幅。指定された場合、scale_factorは自動計算される
            resampling: リサンプリング方法（デフォルト: bilinear）

        Returns:
            np.ndarray: 縮小された画像データ（BGR形式、shape: (height, width, channels)）

        Raises:
            ValueError: scale_factorとtarget_widthの両方が未指定、または不正な値の場合

        Note:
            この関数は元のTIFFファイルを変更しません。読み取り専用モードでも動作します。
            rasterioのout_shapeパラメータを使用してメモリ効率的に縮小を行います。
        """
        if scale_factor is None and target_width is None:
            raise ValueError(
                "scale_factorまたはtarget_widthのいずれかを指定してください"
            )

        # 元画像の情報を取得
        original_height, original_width, channels = self.shape

        # スケールファクターの計算
        if target_width is not None:
            if target_width <= 0 or target_width > original_width:
                raise ValueError(
                    f"target_widthは1以上{original_width}以下である必要があります"
                )
            scale_factor = target_width / original_width
        else:
            if scale_factor <= 0 or scale_factor > 1:
                raise ValueError("scale_factorは0より大きく1以下である必要があります")

        # 出力サイズの計算
        output_width = int(original_width * scale_factor)
        output_height = int(original_height * scale_factor)

        self.logger.info(
            f"画像を縮小中: {original_width}x{original_height} -> {output_width}x{output_height} (scale: {scale_factor:.3f})"
        )

        # rasterioハンドルの確認（読み取り専用でも可）
        handle = self._rasterio_handle or self._tiff_handle
        if not handle:
            raise ValueError("ファイルハンドルが利用できません")

        if self._rasterio_handle:
            # rasterioハンドルを使用してメモリ効率的に縮小
            # out_shapeパラメータを使用して読み込み時に縮小
            scaled_data = self._rasterio_handle.read(
                out_shape=(channels, output_height, output_width), resampling=resampling
            )
        else:
            # tifffileハンドルの場合は全体を読み込んでからリサイズ
            # （この場合はメモリ効率が劣るが、読み取り専用で動作）
            page = self._tiff_handle.pages[0]
            full_data = page.asarray()

            if full_data.ndim == 3:
                # (height, width, channels) -> (channels, height, width)
                full_data = np.transpose(full_data, (2, 0, 1))
            else:
                # 単一チャンネルの場合
                full_data = full_data[np.newaxis, :, :]
                channels = 1

            # OpenCVを使用してリサイズ
            scaled_data = np.zeros(
                (channels, output_height, output_width), dtype=full_data.dtype
            )
            for i in range(channels):
                scaled_data[i] = cv2.resize(
                    full_data[i],
                    (output_width, output_height),
                    interpolation=(
                        cv2.INTER_LINEAR
                        if resampling == Resampling.bilinear
                        else cv2.INTER_NEAREST
                    ),
                )

        # チャンネル順を変更: (channels, height, width) -> (height, width, channels)
        if channels > 1:
            scaled_data = np.transpose(scaled_data, (1, 2, 0))
        else:
            scaled_data = scaled_data[0]  # 単一チャンネルの場合

        # TIFFファイルはRGB形式で保存されているため、BGR形式に変換（OpenCV互換）
        if channels == 3:
            scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_RGBA2BGRA)

        self.logger.info(f"縮小完了（BGR形式）: 出力形状 {scaled_data.shape}")
        return scaled_data


def create_sample_tiff(filepath: str, shape: Tuple[int, int, int], tilesize: int = 512):
    """サンプルのタイル化TIFFファイルを作成する関数（RGB形式でTIFFに保存）"""
    height, width, channels = shape

    # グラデーションパターンを作成
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    if channels == 3:
        # カラーグラデーション（RGB順序で作成）
        r = (x_coords / width * 255).astype(np.uint8)
        g = (y_coords / height * 255).astype(np.uint8)
        b = ((x_coords + y_coords) / (width + height) * 255).astype(np.uint8)
        data = np.stack([r, g, b], axis=2)
    else:
        # グレースケールグラデーション
        data = ((x_coords + y_coords) / (width + height) * 255).astype(np.uint8)

    # タイル化TIFFとして保存（RGB形式）
    tifffile.imwrite(
        filepath,
        data,
        tile=(tilesize, tilesize),
        photometric="rgb" if channels == 3 else "minisblack",
    )

    return filepath


def test_tiff_editor():
    """TiffEditorのテスト関数"""
    import tempfile
    import cv2

    logging.basicConfig(level=logging.DEBUG)

    # テスト用の一時ファイル
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        temp_filepath = tmp.name

    try:
        # サンプルTIFFファイルを作成
        print("サンプルTIFFファイルを作成中...")
        create_sample_tiff(temp_filepath, (2000, 3000, 3), tilesize=256)

        # TiffEditorでファイルを開く
        with TiffEditor(temp_filepath, mode="r+") as editor:
            print(f"ファイル情報: {editor.get_info()}")

            # 部分的に読み込み
            print("部分読み込みテスト...")
            region_data = editor[100:300, 200:400]  # 200x200の領域
            print(f"読み込んだ領域の形状: {region_data.shape}")

            # 読み込んだ領域を変更
            print("部分書き込みテスト...")
            modified_data = np.zeros_like(region_data)
            modified_data[:, :, 0] = 255  # 赤色にする
            editor[100:300, 200:400] = modified_data

            # 変更を確認
            print("変更確認...")
            verification_data = editor[100:300, 200:400]
            print(f"変更後の平均値 (R,G,B): {np.mean(verification_data, axis=(0,1))}")

            # 別の領域を読み込んで表示用に保存（TiffEditorはBGR形式を返すのでそのまま保存）
            display_region = editor[0:500, 0:500]
            cv2.imwrite(
                "tiff_editor_test_output.png",
                display_region,
            )
            print("テスト結果を 'tiff_editor_test_output.png' に保存しました")

        print("TiffEditorのテストが完了しました！")

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)


def test():
    import sys
    import cv2
    import logging

    logging.basicConfig(level=logging.DEBUG)

    png = sys.argv[1]
    tilesize = int(sys.argv[2])

    # 元画像を読み込んでサイズを取得
    original_image = cv2.imread(png)
    if original_image is None:
        print(f"エラー: 画像ファイル '{png}' が見つかりません")
        return

    height, width, channels = original_image.shape
    print(f"元画像サイズ: {height}x{width}x{channels}")

    # TIFFファイルを新規作成（十分な大きさで）
    tiff_height = height + 100  # 余裕を持たせる
    tiff_width = width + 100

    with TiffEditor(
        filepath=png + ".tiff",
        mode="r+",
        tilesize=tilesize,
        shape=(tiff_height, tiff_width, channels),
        dtype=np.uint8,
        create_if_not_exists=True,
    ) as tiff_editor:
        print(f"TIFFファイル情報: {tiff_editor.get_info()}")

        # CV2で読み込んだBGR画像をそのまま使用（TiffEditorはBGR形式を受け取る）
        bgr_image = original_image

        # 画像を配置
        tiff_editor[20 : 20 + height, 40 : 40 + width] = bgr_image
        tiff_editor[10 : 10 + height, 20 : 20 + width] = bgr_image

        # 結果を取得して表示（TiffEditorはBGR形式を返す）
        result_image = tiff_editor[0:tiff_height, 0:tiff_width]

        # 結果はすでにBGR形式なのでそのまま表示
        cv2.imshow("TIFF Editor Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"TIFFファイル '{png}.tiff' が作成されました")


def test_large_tiff():
    """メモリに乗らない大きなTIFFファイルを作成・テストする関数"""
    import psutil
    import gc

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # システムメモリの情報を取得
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    logger.info(f"利用可能メモリ: {available_memory_gb:.2f} GB")

    # メモリより大きなサイズを設定（GB単位で指定）
    # 例：利用可能メモリの1.5倍のサイズの画像を作成
    target_memory_gb = min(available_memory_gb * 1.5, 8.0)  # 最大8GBに制限

    # RGB画像で1ピクセル3バイトとして計算
    bytes_per_pixel = 3
    total_pixels = int(target_memory_gb * 1024**3 / bytes_per_pixel)

    # 正方形に近い形状で計算
    side_length = int(np.sqrt(total_pixels))
    height = width = side_length
    channels = 3

    logger.info(f"作成予定サイズ: {height}x{width}x{channels}")
    logger.info(f"予想ファイルサイズ: {height * width * channels / (1024**3):.2f} GB")

    large_tiff_path = "large_test.tiff"
    tilesize = 512

    try:
        # メモリ使用量を監視しながら大きなTIFFファイルを作成
        logger.info("大きなTIFFファイルを作成中...")
        initial_memory = psutil.Process().memory_info().rss / (1024**2)

        with TiffEditor(
            filepath=large_tiff_path,
            mode="r+",
            tilesize=tilesize,
            shape=(height, width, channels),
            dtype=np.uint8,
            create_if_not_exists=True,
        ) as editor:

            logger.info(f"TIFFファイル情報: {editor.get_info()}")

            # タイルサイズで分割して段階的に書き込み
            tile_h, tile_w = tilesize, tilesize
            tiles_written = 0
            total_tiles = (height // tile_h + 1) * (width // tile_w + 1)

            for y in range(0, height, tile_h):
                for x in range(0, width, tile_w):
                    # 実際のタイル範囲を計算
                    y_end = min(y + tile_h, height)
                    x_end = min(x + tile_w, width)
                    actual_tile_h = y_end - y
                    actual_tile_w = x_end - x

                    # パターン化されたタイルデータを作成
                    tile_data = create_pattern_tile(
                        actual_tile_h, actual_tile_w, channels, x, y
                    )

                    # タイルを書き込み
                    editor[y:y_end, x:x_end] = tile_data

                    tiles_written += 1
                    if tiles_written % 100 == 0:
                        current_memory = psutil.Process().memory_info().rss / (1024**2)
                        logger.info(
                            f"タイル進捗: {tiles_written}/{total_tiles} "
                            f"(メモリ使用量: {current_memory:.1f}MB)"
                        )
                        gc.collect()  # ガベージコレクション

            final_memory = psutil.Process().memory_info().rss / (1024**2)
            logger.info(
                f"メモリ使用量変化: {initial_memory:.1f}MB -> {final_memory:.1f}MB"
            )

        # ファイルが正しく作成されたかチェック
        logger.info("ファイル整合性チェック中...")
        consistency_check_passed = check_tiff_consistency(large_tiff_path, tilesize)

        if consistency_check_passed:
            logger.info("✅ 大きなTIFFファイルの作成・整合性チェックが成功しました！")
        else:
            logger.error("❌ ファイル整合性チェックに失敗しました")

        # ファイルサイズ情報
        actual_size_gb = os.path.getsize(large_tiff_path) / (1024**3)
        logger.info(f"実際のファイルサイズ: {actual_size_gb:.2f} GB")

    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        raise
    finally:
        # クリーンアップ
        if os.path.exists(large_tiff_path):
            logger.info(f"テストファイル {large_tiff_path} を削除中...")
            os.unlink(large_tiff_path)


def create_pattern_tile(
    height: int, width: int, channels: int, offset_x: int, offset_y: int
) -> np.ndarray:
    """パターン化されたタイルデータを作成"""
    if channels == 3:
        # 座標に基づくカラーパターン
        r = ((offset_x % 256) * np.ones((height, width))).astype(np.uint8)
        g = ((offset_y % 256) * np.ones((height, width))).astype(np.uint8)
        b = (((offset_x + offset_y) % 256) * np.ones((height, width))).astype(np.uint8)
        return np.stack([r, g, b], axis=2)
    else:
        # グレースケールパターン
        return (((offset_x + offset_y) % 256) * np.ones((height, width))).astype(
            np.uint8
        )


def check_tiff_consistency(filepath: str, tilesize: int) -> bool:
    """TIFFファイルの整合性をチェック"""
    logger = logging.getLogger(__name__)

    try:
        with TiffEditor(filepath, mode="r") as editor:
            height, width, channels = editor.shape
            logger.info(f"読み込んだファイル形状: {height}x{width}x{channels}")

            # ランダムな位置のタイルをいくつかサンプリングしてチェック
            import random

            sample_count = min(10, (height // tilesize) * (width // tilesize))

            for i in range(sample_count):
                # ランダムなタイル位置を選択
                tile_y = random.randint(0, height // tilesize) * tilesize
                tile_x = random.randint(0, width // tilesize) * tilesize

                tile_y_end = min(tile_y + tilesize, height)
                tile_x_end = min(tile_x + tilesize, width)

                # タイルを読み込み
                tile_data = editor[tile_y:tile_y_end, tile_x:tile_x_end]

                # 期待値と比較
                expected_tile = create_pattern_tile(
                    tile_y_end - tile_y, tile_x_end - tile_x, channels, tile_x, tile_y
                )

                if not np.array_equal(tile_data, expected_tile):
                    logger.error(f"タイル ({tile_x}, {tile_y}) のデータが一致しません")
                    return False

                logger.debug(f"タイル ({tile_x}, {tile_y}) の整合性チェック完了")

            logger.info(f"整合性チェック完了: {sample_count}個のタイルをテストしました")
            return True

    except Exception as e:
        logger.error(f"整合性チェック中にエラー: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "large_test":
        test_large_tiff()
    elif len(sys.argv) > 1 and sys.argv[1] == "test_editor":
        test_tiff_editor()
    else:
        test()
