# es_physics

進化戦略（ES）と誤差逆伝播（BP）の等価性（Stein の補題）を出発点に、
「非同期に更新される零和エージェントのネットワーク」という言葉で物理をどこまで読めるかを
数値実験で探索した研究ノートです。

- [ES_Infinite_Dimensional_Backpropagation.pdf](ES_Infinite_Dimensional_Backpropagation.pdf) — 出発点の論文（ES = BP、Stein の補題、2026-01）
- [es_physics_notes.html](es_physics_notes.html) — 時空と重力（虚数単位 i の出どころ、光円錐、質量、重力、重力波、時間が 1 本である理由）
- [es_forces_notes.html](es_forces_notes.html) — 力（光子、閉じ込め、ES = BP としての勾配フロー、弱い力、中性子の寿命）
- [es_ct_notes.html](es_ct_notes.html) — 光速 × 宇宙の年齢 = 一定（規約と主張の境目、共動座標・ネットワークの時計・ES のステップ幅の対応）
- [es_ct_summary.html](es_ct_summary.html) — 上のノートの要約（ここまでにわかったこと、観測との突き合わせ、訂正の記録）
- [koide_notes.html](koide_notes.html) — Koide 公式（規則を計算前に固定し、クォーク・ニュートリノで検定 → 棄却）
- [scripts/](https://github.com/yomei-o/es_physics/tree/main/scripts) — 各節の数値実験（Python 3 + numpy / scipy）

Web で読む：https://yomei-o.github.io/es_physics/

ブラウザで HTML を開き、印刷（Ctrl+P）で PDF にできます（数式は KaTeX を CDN から読み込みます）。

各節は実行前に予測を書いてから計算し、「検証済み／解釈／未解決・否定的」を区別しています。
多くの結果は既知の物理の再構成であり、外れた予測ややり直しもそのまま記録しています。
