# PNT-simulation

GNSS denied/degraded 環境を想定し、**Iridium系PNTによる時刻拘束**と、**既存LEO通信衛星のDoppler観測による自己位置推定**を組み合わせるための研究用Pythonシミュレータです。

## Ver.1 の目的

GNSS喪失直後から、次の4ケースを同一条件で比較します。

| Case | Iridium timing | LEO Doppler | 意味 |
|---|---:|---:|---|
| A | OFF | OFF | 受信機時計・状態の自走 |
| B | ON | OFF | 時刻バックアップのみ |
| C | OFF | ON | LEO Doppler測位のみ |
| D | ON | ON | Hybrid LEO-PNT |

Ver.1 は「物理層の通信波形」を再現するものではなく、観測値レベルのシミュレーションです。

- LEO衛星: 円軌道近似
- 受信機: 地上固定
- LEO観測: Doppler
- Iridium観測: 受信機時計バイアス
- 推定器: 8状態 Extended Kalman Filter (EKF)
- 出力: 位置誤差、時刻誤差、可視LEO衛星数

状態ベクトルは

```text
[x, y, z, vx, vy, vz, clock_bias, clock_drift]
```

です。位置・速度はECEF、時計バイアスは秒、時計ドリフトは秒/秒で扱います。

## セットアップ

Python 3.10+ を推奨します。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 実行

```bash
python main.py
```

`output/` に以下を生成します。

- `position_error.png`
- `clock_error.png`
- `visible_satellites.png`
- `summary.csv`

## 主要パラメータ

`config.py` で変更できます。

- シミュレーション時間
- Iridium時刻観測精度
- LEO Doppler観測雑音
- LEO衛星数
- 搬送波周波数
- 仰角マスク
- 初期位置・時刻誤差
- 受信機時計のランダムウォーク

## 注意

このVer.1は研究仮説を検証するための骨格です。Starlink/Iridiumの実際の非協調測位では、衛星エフェメリス誤差、CFO、衛星時計、ビーム切替、信号構造、追尾性能などが支配的になる可能性があります。これらは今後のモデル拡張対象です。

## 今後の拡張候補

1. TLE + SGP4による実衛星軌道
2. Starlink/OneWeb/Iridium/Orbcommの個別モデル
3. 衛星エフェメリス誤差
4. CFO・受信機発振器モデル
5. 移動受信機
6. INSとの統合
7. TOA/TDOA/Carrier phase観測
8. Monte Carlo評価
9. GNSS妨害開始時刻を含むシナリオ
10. 実SDRログとの比較

## 研究上の中心的な問い

> GNSSが利用できない環境で、Iridium系PNTによって時刻を拘束した場合、既存LEO通信衛星のDoppler測位精度と収束性がどの程度改善するか。
