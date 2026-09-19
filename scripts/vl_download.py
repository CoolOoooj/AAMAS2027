from datasets import load_dataset
ds = load_dataset("MMInstruction/VL-RewardBench")
ds.save_to_disk("/mnt/dataset3/hongjun/ERMJ/data/VL-RewardBench")
print("数据集已成功保存到本地！")