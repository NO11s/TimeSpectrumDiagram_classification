# 智能硬件语音控制的时频图分类挑战赛

## GPU资源

```shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.48.07    Driver Version: 515.48.07    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
| N/A   29C    P0    23W / 250W |      8MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## 依赖

```shell
python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pandas
pip install matplotlib
```

## 代码框架

1. 数据处理

   ```
   /dataset.py
   ```
2. 模型设计：[weiaicunzai/pytorch-cifar100: Practice on cifar100](https://github.com/weiaicunzai/pytorch-cifar100)

   ```
   /models/resnet.py
   /models/senet.py
   ```
3. 模型训练（包括验证集的随机划分）

   ```
   /train.py
   ```
4. 模型测试

   ```
   /infer.py
   ```
5. 训练过程可视化

   ```
   /visualize.py
   ```

## TODO

- [ ] 图像数据预处理/数据增强
- [X] 标签数据预处理
- [X] 训练集的划分（按照类别等概率）
- [X] 训练过程可视化
- [ ] 测试不同的优化器
- [ ] 模型优化
- [ ] 交叉验证
