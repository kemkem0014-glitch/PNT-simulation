# 遮蔽環境における天空開口部配置が単一LEO衛星Doppler測位に与える影響

## 中心仮説

森林・都市峡谷等の遮蔽環境では、単一LEO衛星が複数の小さな天空開口部を順次横切る。総観測時間または総開口角が同じでも、1個の連続開口より、時間的に分散した複数開口の方がsingle-LEO Doppler測位のgeometry diversityを高める可能性がある。

## 観測式

```math
\dot{\rho}(t)=\frac{(\mathbf{s}(t)-\mathbf{r})^T\mathbf{v}_s(t)}{\|\mathbf{s}(t)-\mathbf{r}\|}+b_{\dot{\rho}}+\varepsilon
```

```math
f_D(t)=-\frac{f_c}{c}\dot{\rho}(t)
```

未知量は既知高度を仮定して

```math
\theta=[x,y,b_{\dot{\rho}}]^T
```

とする。FIM/CRLBは

```math
\mathbf{J}=\sum_{k\in\mathcal{T}}\mathbf{H}_k^T\mathbf{R}^{-1}\mathbf{H}_k,
\qquad
\mathbf{P}_{CRLB}=\mathbf{J}^{-1}
```

で評価する。

## 基準条件

- LEO高度 550 km
- cross-track offset 300 km
- 最大仰角 約61.39 deg
- elevation mask 20 deg
- carrier 2 GHz
- range-rate noise 0.2 m/s
- 1 Hz
- static receiver / known altitude
- common range-rate bias unknown

## 実験A: 総観測時間30秒一定

|開口数|各開口時間[s]|最適CRLB水平1σ[m]|ランダム配置中央値[m]|MC RMS[m]|
|---:|---:|---:|---:|---:|
|1|30.0|220.9|2929.9|209.1|
|2|15.0|63.6|228.9|63.3|
|3|10.0|22.4|49.4|21.9|
|4|7.5|22.0|33.1|21.8|
|5|6.0|21.1|29.6|-|
|6|5.0|20.6|27.7|-|
|8|3.75|21.8|27.0|21.7|
|10|3.0|19.4|26.2|-|

最大の改善は1→2→3開口で生じ、その後は20 m級で改善が飽和する。

### 最適3開口

TCA=0として、総観測30秒を3×10秒へ分割した最適中心は以下。

- -72 s: az 298.8 deg, el 41.4 deg, effective track width 3.94 deg
- 0 s: az 0 deg, el 61.4 deg, effective track width 6.93 deg
- +72 s: az 61.2 deg, el 41.4 deg, effective track width 3.94 deg

同じ10秒でも、TCA付近は見かけ角速度が大きいため必要な物理的開口角が大きい。

## 実験B: 総開口角30 deg一定

|開口数|各開口角[deg]|観測サンプル数|CRLB水平1σ[m]|
|---:|---:|---:|---:|
|1|30.0|45|87.3|
|2|15.0|68|25.4|
|3|10.0|86|14.4|
|4|7.5|112|13.8|
|5|6.0|102|13.7|
|6|5.0|100|13.5|
|8|3.75|112|13.2|
|10|3.0|114|13.4|

## 理論的な開口最適化

開口集合Aに対して

```math
\mathcal{A}^{*}=\arg\min_{\mathcal{A}}\operatorname{tr}(\mathbf{S}\mathbf{J}(\mathcal{A})^{-1}\mathbf{S}^T)
```

と定式化できる。

ただし「絶対的な最適開口サイズ」はacquisition/reacquisition時間、最低連続受信時間、C/N0、仰角依存雑音等の制約なしには一意に定まらない。今後は最低連続捕捉時間を工学制約として追加する。

## 先行研究との差

single-LEO Doppler positioning、time-diverse Doppler-only PNT、同一衛星の複数時刻をfaux satellitesとして扱う考え方は既出である。またLEO通信で魚眼画像からclear/shadowed/blockedを予測する研究、Starlink toneのintermittent trackingをmulti-epoch測位へ集約する研究もある。

本研究の狙いは、**environment-induced sky-aperture geometry → intermittent observation timing → Doppler information geometry** を明示し、開口の数・大きさ・配置そのものを測位設計変数として評価する点にある。

## 参考文献

1. Q. Liu et al., “Geometric Performance Analysis of Doppler-Based Positioning with a Single LEO Satellite,” arXiv:2603.19499, 2026.
2. M. O. Moore et al., “Time-Diverse Doppler-only LEO PNT,” MILCOM 2023, doi:10.1109/MILCOM58377.2023.10356223.
3. M. O. Moore et al., “Analysis of Time-Diverse Doppler-Only LEO PNT,” IEEE/ION PLANS 2025, doi:10.1109/PLANS61210.2025.11028336.
4. K.-M. Cheung et al., “Single-Satellite Doppler Localization with Law of Cosines (LOC),” IEEE Aerospace Conference, 2019, doi:10.1109/AERO.2019.8742181.
5. R. Akturan and W. Vogel, “Path diversity for LEO satellite-PCS in the urban environment,” IEEE TAP, 1997, doi:10.1109/8.596901.
6. N. Jardak and R. Adam, “Practical Use of Starlink Downlink Tones for Positioning,” Sensors 23, 3234, 2023, doi:10.3390/s23063234.
