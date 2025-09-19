# Rasterio TIFF Editor

メモリ効率の良い巨大 TIFF ファイルの部分編集を可能にする Python ライブラリです。

## 特徴

- 🚀 **メモリ効率**: メモリに乗らない巨大な TIFF ファイルでも部分的な読み書きが可能
- 🔧 **タイル構造**: TIFF のタイル機能を活用した高速アクセス
- 🎯 **スライス記法**: NumPy ライクなスライス記法でのデータアクセス
- 📊 **データ整合性**: 読み書きの整合性チェック機能内蔵
- 💾 **部分更新**: 既存ファイルの一部のみを効率的に更新可能

## インストール

### 別のプロジェクトで使用する場合（推奨）

pipでGitHubから直接インストール：

```bash
pip install git+https://github.com/vitroid/rasterio_tiff.git
```

またはPoetryを使用している場合：

```bash
poetry add git+https://github.com/vitroid/rasterio_tiff.git
```

### 開発環境として使用する場合

```bash
# プロジェクトをクローン
git clone https://github.com/vitroid/rasterio_tiff.git
cd rasterio_tiff

# 依存関係をインストール
poetry install

# 仮想環境をアクティブ化
poetry shell
```

## 基本的な使用方法

### 新しい TIFF ファイルの作成

```python
from rasterio_tiff import TiffEditor
import numpy as np

# 新しいTIFFファイルを作成
with TiffEditor(
    filepath="large_image.tiff",
    mode="r+",
    tilesize=512,
    shape=(10000, 10000, 3),  # 高さ×幅×チャンネル
    dtype=np.uint8,
    create_if_not_exists=True,
) as editor:
    # 部分的にデータを書き込み
    test_data = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    editor[1000:2000, 2000:3000] = test_data

    print(f"ファイル情報: {editor.get_info()}")
```

### 既存 TIFF ファイルの読み書き

```python
from rasterio_tiff import TiffEditor

# 既存ファイルを開く
with TiffEditor("existing_image.tiff", mode="r+") as editor:
    # 部分的にデータを読み込み
    region_data = editor[500:1500, 1000:2000]  # 1000x1000の領域
    print(f"読み込んだ領域の形状: {region_data.shape}")

    # データを変更して書き戻し
    modified_data = region_data.copy()
    modified_data[:, :, 0] = 255  # 赤チャンネルを最大に
    editor[500:1500, 1000:2000] = modified_data
```

## パフォーマンステスト

メモリに乗らない大きなファイルのテストも可能です：

```bash
# 開発環境の場合
poetry run python rasterio_tiff/tiffeditor.py large_test

# または直接Pythonで実行
python -c "from rasterio_tiff.tiffeditor import test_large_tiff; test_large_tiff()"
```

このテストでは以下が確認されます：

- メモリ使用量の監視
- ファイルサイズ vs メモリ使用量の効率
- データ整合性の検証
- タイル構造の正常動作

## テスト例の結果

```
INFO:__main__:利用可能メモリ: 4.24 GB
INFO:__main__:作成予定サイズ: 47717x47717x3
INFO:__main__:予想ファイルサイズ: 6.36 GB
INFO:__main__:実際のファイルサイズ: 6.47 GB
INFO:__main__:メモリ使用量変化: 52.8MB -> 1311.8MB
INFO:__main__:✅ 大きなTIFFファイルの作成・整合性チェックが成功しました！
```

**メモリに乗らないサイズ（6.47GB）のファイルを、わずか 1.31GB のメモリで処理！**

## 技術仕様

- **依存関係**: numpy, rasterio, tifffile, opencv-python
- **対応形式**: タイル化 TIFF（Tiled TIFF）
- **データ型**: uint8, uint16, float32 など（NumPy 対応型）
- **最大ファイルサイズ**: システムディスク容量に依存
- **推奨タイルサイズ**: 256, 512, 1024 ピクセル

## クラス構造

### `TiffEditor`

メインクラス。TIFF ファイルの読み書きを担当。

### `Rect`, `Range`

2 次元領域と 1 次元範囲を表現するデータクラス。

### 主要メソッド

- `get_region(region: Rect)`: 指定領域のデータを取得
- `put_region(region: Rect, data: np.ndarray)`: 指定領域にデータを書き込み
- `get_info()`: ファイル情報を取得
- `__getitem__`, `__setitem__`: スライス記法でのアクセス

## ライセンス

MIT License

## 開発者向け

### 既存テスト関数

```bash
# 基本的なエディタテスト
poetry run python rasterio_tiff/tiffeditor.py test_editor

# 画像ファイルからTIFF作成テスト
poetry run python rasterio_tiff/tiffeditor.py your_image.png 512

# または関数を直接インポートして実行
python -c "from rasterio_tiff.tiffeditor import test_tiff_editor; test_tiff_editor()"
```

### パフォーマンス要件

- メモリ使用量: ファイルサイズの 20-30%程度
- タイルアクセス: O(1)の複雑度
- 同時書き込み: 単一プロセス推奨

---

**注意**: このライブラリは大きなファイルを扱うため、十分なディスク容量があることを確認してください。
