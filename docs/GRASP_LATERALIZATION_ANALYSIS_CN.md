# GRASP 左右侧化分析规范

## 核心原则

GRASP 中出现部分果蝇右侧更强、部分左侧更强，并不自动支持或否定群体右偏。分析必须同时回答：

1. 有向均值 `(R-L)/(R+L)` 的置信区间是否跨 0；
2. 右偏与左偏个体的比例是否偏离 1:1；
3. 绝对侧化 `abs((R-L)/(R+L))` 是否超过阴性对照或重复测量噪声；
4. 方向和强度是否受 batch、脑朝向、曝光或 ROI 规则影响。

若有向均值跨 0，但绝对侧化显著超过阴性对照，结论是“方向在个体间可变的局部侧化”。若没有阴性对照或重复测量，绝对左右差只能描述为方向异质性，不能单独证明生物侧化。

## 最小输入表

每行可以是一只果蝇，也可以是同一只果蝇的一次技术重复：

```csv
fly_id,group,batch,left_signal,right_signal,left_background,right_background,exclude
fly_001,experimental,batch_1,120.4,88.1,10.2,9.7,false
fly_002,experimental,batch_1,75.2,111.6,8.9,9.1,false
ctrl_001,control,batch_1,92.0,90.5,9.4,9.0,false
```

必须列：`fly_id,left_signal,right_signal`。建议列：`group,batch,left_background,right_background,exclude`。左右必须是按真实脑侧配准后的值，不能直接使用相机画面左右。

## 运行

```bash
PYTHONPATH=simulation/src python simulation/scripts/analyze_grasp_lateralization.py \
  --input /absolute/path/to/grasp_measurements.csv \
  --output-dir outputs/grasp_lateralization
```

输出：

- `grasp_fly_level.csv`：fly-level 左右信号、signed LI、absolute LI 和方向；
- `grasp_batch_summary.csv`：批次层级有向与绝对侧化；
- `grasp_statistics.json`：bootstrap、sign-flip、sign test 和对照置换统计；
- `Fig_grasp_lateralization.pdf/png`：配对信号、个体方向和绝对侧化强度。

## 论文措辞决策

- signed LI 的 95% CI 全部大于 0：可写“population-average right shift with individual variability”，不能写“all flies are right-lateralized”。
- signed LI 的 CI 跨 0，absolute LI 高于 control：写“direction-variable lateralization”，不指定群体固定方向。
- signed LI 的 CI 跨 0，且没有 control：只写“individual left-right heterogeneity”，等待阴性对照或重复测量。
- 不论结果如何，单个 FlyWire 脑中的右偏方向都只能称为 reference-connectome direction，不能外推为果蝇物种固定右偏。
