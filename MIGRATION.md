# 🚀 プロジェクト移行のお知らせ

## 新しいリポジトリへの移行

このプロジェクトは以下の新しいリポジトリに移行しました：

**🆕 新しいリポジトリ: [`vitroid/tiffeditor`](https://github.com/vitroid/tiffeditor)**

### なぜ移行したのか？

- **より適切な名前**: `tiffeditor`はプロジェクトの機能をより正確に表現
- **依存関係の明確化**: `rasterio`との混同を避け、独立したライブラリとして明確化
- **将来性**: より拡張しやすい名前空間

### 既存ユーザーへの影響

#### ✅ **互換性維持**
- このリポジトリ（`rasterio_tiff`）は**引き続き利用可能**
- 既存のインストール方法は変更なし
- パッケージ名 `rasterio_tiff` は維持

#### 🔄 **新しいインストール方法（推奨）**

```bash
# 新しいリポジトリから（推奨）
pip install git+https://github.com/vitroid/tiffeditor.git

# 従来通り（互換性のため残存）
pip install git+https://github.com/vitroid/rasterio_tiff.git
```

#### 📦 **パッケージ名の変更**

新しいリポジトリでは、以下のように使用してください：

```python
# 新しい方法（推奨）
from tiffeditor import TiffEditor, ScalableTiffEditor

# 従来の方法（互換性のため残存）
from rasterio_tiff import TiffEditor, ScalableTiffEditor
```

### 移行スケジュール

- **現在〜2024年末**: 両方のリポジトリを並行維持
- **2025年1月〜**: 新機能は`tiffeditor`リポジトリのみ
- **2025年中**: `rasterio_tiff`は重要なバグフィックスのみ

### 最新機能

新しいリポジトリ（`tiffeditor`）でのみ利用可能：

- ✨ **ScalableTiffEditor**: 仮想的な大画像操作
- 🎨 **BGR形式統一**: OpenCV完全互換
- ⚡ **パフォーマンス改善**: ロガー最適化
- 🧪 **充実したテストスイート**

### アクション推奨事項

1. **新規プロジェクト**: `vitroid/tiffeditor` を使用
2. **既存プロジェクト**: 段階的に移行を検討
3. **問題報告**: どちらのリポジトリでも受付

---

ご質問やサポートが必要な場合は、[Issues](https://github.com/vitroid/tiffeditor/issues)でお気軽にお問い合わせください。
