# 遮蔽環境における Starlink Direct-to-Cell ドップラー測位の理想化天空開口シミュレーション

日本航海学会論文集（JIN）投稿原稿
「天空が限られた遮蔽環境におけるStarlink Direct-to-Cellを用いたドップラー測位の到達精度
－理想化天空開口モデルによる事前シミュレーション－」の計算コードと結果一式です。

携帯電話が通信に使う Starlink Direct-to-Cell（DTC）衛星を、そのまま測位にも使えないかを検討する前段として、
天空の一部しか見えない環境を「直径10°の円形開口がN個ある」と理想化し、ドップラー（距離変化率）観測で
得られる位置精度の下限（CRLB）と、それに必要な待ち時間を評価します。

## 計算条件（論文3章）

| 項目 | 設定 |
|---|---|
| 衛星 | CelesTrak GP/OMM（2026-10-06保存）のうち名称に[DTC]を含む637機（傾斜角43°殻327機、53°殻310機） |
| 軌道計算 | SGP4 → TEME座標 → GMST回転でECEF |
| 受信点 | 東京海洋大学 越中島 RTK基準点（35.6662430056°N, 139.7923087°E, 楕円体高59.9 m）、静止 |
| 観測 | 1秒ごとの距離変化率、雑音 σ = 0.2 m/s、衛星ごとの未知バイアスあり |
| 未知量 | 受信点の3次元位置＋衛星ごとのバイアス（バイアスはシューア補行列で消去） |
| 開口 | 直径10°の円形開口 N = 1～6。基準配置は N≥3 で仰角35.264°・等間隔方位、N=2 は仰角45°の対向、N=1 は方位0°・仰角22.5° |
| 評価窓 | 開始時刻 00:00～23:00 を120秒ごと（691通り）× 待ち時間 1～60分 |
| 評価量 | 水平・垂直・3次元CRLBの中央値とP90、達成率 A_h(T)、所要待ち時間 T_{h,q}（例：T_{100,0.9}） |
| 有効窓 | 逆行列が数値的に安定（最小固有値 > 1e-18 かつ 条件数 < 1e12）かつ 3次元CRLB ≤ 1 km |

## ファイル構成

### 論文用（現行）

| ファイル | 役割 |
|---|---|
| `make_fig1.py` | **図1**を単独で作成。OMMファイル → SGP4 → 方位・仰角 → 開口方向ごとの24時間観測点数 → 白黒の天空図（N=1～6） |
| `revision_common.py` | 共通部品。2日分・1秒刻みの衛星幾何キャッシュの作成と読み込み、開口による観測点の選別、フィッシャー情報行列とCRLBの計算 |
| `revision_experiments_20261012.py` | 論文4章の全実験（E0～E8）。結果を `results/revision_20261012/*.csv` に出力 |
| `figures_jin_20261013.py` | **図2・図3**（JIN段幅80 mm、白黒印刷向け）を作成 |
| `sky_density_N1_20261015.py` | 単一開口（N=1）を置く方向ごとの観測点数・衛星数（`E9_sky_density_N1.csv`） |
| `reproduce_3d_paper_20261006.py` | 保存済みOMMからの3次元CRLB再計算（旧原稿表8～12）。`revision_common.py` が軌道計算の規約をここから読み込む |
| `validate_reproduced_3d.py` | 上記結果の独立な検算（バイアスを直接中心化して比較） |

### 結果フォルダ `results/`

| フォルダ・ファイル | 内容 |
|---|---|
| `tumsat_real_dtc_20261006/dtc_omm_snapshot.csv` | **入力データ**。計算に使った軌道要素（2026-10-06保存）。すべての計算はこのファイルから再現できます |
| `revision_20261012/E0_baseline_all_waits.csv` | 基準配置 N=1～6、待ち時間1～60分、1日目・2日目（表1・表2・図2） |
| `revision_20261012/E0_T_hq.csv` | 所要待ち時間 T_{h,q}（表1） |
| `revision_20261012/E2_azimuth_sweep.csv` | 開始方位を5°刻みで回したときの結果（表3・図3） |
| `revision_20261012/E1_nested*.csv` | 開口を1個ずつ追加したときの変化（表4） |
| `revision_20261012/E3_grid_search.csv` | 仰角×開始方位の探索（1日目で選定、2日目で評価）（表5） |
| `revision_20261012/E4_same_elevation_concentrated.csv`, `E4b_*.csv` | 同じ仰角での集中配置と分散配置の比較（表6） |
| `revision_20261012/E6_linear_drift.csv` | 未知の周波数ドリフトを加えた場合（表7） |
| `revision_20261012/E7_orbit_error.csv` | 衛星位置・速度の誤差を加えた場合（表7） |
| `revision_20261012/E8_gauss_newton_mc.csv` | Gauss–Newton推定のモンテカルロ確認（表8） |
| `revision_20261012/E9_sky_density_N1.csv` | 単一開口の方向別観測点数 |
| `revision_20261012/jin/` | 論文用の図（`fig1_sky_density.png`＝図1、`jin_fig2_wait.png`＝図2、`jin_fig3_azimuth.png`＝図3）と図1の元データ `fig1_sky_density.csv`。`jin_fig1_sky.png` は `figures_jin_20261013.py` がキャッシュから描く同じ図の旧版 |
| `revision_20261012/metadata.json` | 入力ファイルのSHA-256、キャッシュ点数、有効判定の閾値など |
| `revision_20261012/CHATGPT_TABLE_CHECK.md` | 旧原稿の表13～18と再計算値の照合記録 |
| `reproduced_3d_20261006/` | 旧原稿の表8～12の再計算結果と検算 |

`geometry_cache_2days.npz`（約325 MB）は容量が大きいため GitHub には含めていません。初回実行時に自動で作成されます。

## 実行方法

Python 3.10以上を推奨します。日本語の図には Noto Sans CJK JP フォントを使います。

```bash
cd research/sky_aperture_leo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python make_fig1.py                        # 図1（約15秒）
python revision_experiments_20261012.py    # 表1～8の元データ（初回はキャッシュ作成を含む）
python figures_jin_20261013.py             # 図2・図3
```

E8（Gauss–Newton）は乱数のシードを固定しているため、すべての結果は毎回同じになります。

## 注意

- 開口は円形・一定の大きさ・常時開放と理想化しており、実際の建物や樹木の形状、マルチパス、信号強度による観測の欠落は含みません。
- 観測雑音は全衛星・全仰角で一定（0.2 m/s）としています。
- CRLBは推定精度の下限であり、実際の受信機で得られる精度を保証するものではありません。

## 旧版（参考）

- `aperture_leo_sim.py`、`REPORT.md`、`results/*.csv`（`fixed_total_*`, `optimal_aperture_geometry.csv`）：
  単一の模擬衛星（高度550 km の円軌道）で開口の数・大きさ・配置を比較した初期検討です。
  実行：`python aperture_leo_sim.py --output-dir output_aperture_leo`
- `real_dtc_tumsat.py`、`results/tumsat_real_dtc_20261006/`：CelesTrakから実DTC衛星の軌道要素を取得し、
  3開口の水平CRLBと待ち時間を評価した中間版です（取得した軌道要素は現行計算の入力として使用）。
- `figures_revision_20261012.py`：論文改訂時のカラー版の図（現在は `figures_jin_20261013.py` に置き換え）。
