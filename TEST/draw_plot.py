import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 讀取 result.csv
df = pd.read_csv('result.csv')

# --------- 畫 IOU ---------
plt.figure(figsize=(16, 6))
bars = plt.bar(df['id'], df['iou'], color='skyblue', label='IOU')
avg = df['iou'].mean()
plt.axhline(avg, color='red', linestyle='--', label=f'Avg: {avg:.3f}')
plt.ylabel('IOU')
plt.title('IOU Score by Case')
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig('iou_bar.png')
plt.close()

# --------- 畫 F1 ---------
plt.figure(figsize=(16, 6))
bars = plt.bar(df['id'], df['f1'], color='orange', label='F1')
avg = df['f1'].mean()
plt.axhline(avg, color='red', linestyle='--', label=f'Avg: {avg:.3f}')
plt.ylabel('F1 Score')
plt.title('F1 Score by Case')
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig('f1_bar.png')
plt.close()

print("Bar plots saved as iou_bar.png and f1_bar.png")