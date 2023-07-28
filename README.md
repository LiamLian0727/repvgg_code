# repvgg_code

连仕杰

对RepVGG模型的复现

2023年7月28日

### Training
`python tools/train.py -t config.TrainConfig -m config.RepVGG_A0 -o work_path`

### Test
`python tools/test.py -t config.TrainConfig -m config.RepVGG_A0 -p pth_path`