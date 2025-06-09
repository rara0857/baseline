from tensorboard.backend.event_processing import event_accumulator
import glob, os
import numpy as np


path = '../../LOGS/base40_lvl4_40x_10x/'

os.chdir(path)

file_list = []
# serialize_training_loss = []
serialize_validation_loss = []

# Variable in Training Procedure
batch_size = 16
index_compress_rate = 100
retrieve_mapping = batch_size * index_compress_rate

# Retrieving TF events summary
# Due to the mix of train and test or other events recorded
# Loop through all tfevents files, searching for training events
for filename in sorted(glob.glob("events.out.tfevents.*")):
    file_list.append(filename)
    # print(file_list)
    try:
        flag = True
        event = event_accumulator.EventAccumulator(filename)
        event.Reload()
        for name in event.scalars.Keys():
            # print(name)
            if 'val/val_loss' in name:
                for idx in event.scalars.Items(name):
                    # print(idx)
                    serialize_validation_loss.append(idx.value)
        flag = True
    except:
        continue

    # Tensorboard Exponential Moving Average function


def ema(shadow_variable, variable, decay):
    return decay * shadow_variable + (1 - decay) * variable


# Smoothing value in Tensorboard
smooth = [0.205, 0.369, 0.498, 0.601, 0.683, 0.748, 0.800, 0.841, 0.874, 0.9]  # [0.33,0.5,0.632,0.817,0.999]
candidates = []

for s in smooth:
    large = 99999
    min_ = (0, large, 0)

    # Find Min Value of Loss after Smoothing
    prev = serialize_validation_loss[0]
    for i, v in enumerate(serialize_validation_loss[1:]):
        prev = ema(prev, v, s)  # smooth)
        if prev <= min_[1]:
            min_ = (i + 1, prev, v)

    # Mapping back to the recorded model checkpoint (iteration number) saved
    # The training code record a checkpoint every 8000 iterations
    # 8000/1600 = 5
    target = min_[0]
    model_no = (target // 5) * retrieve_mapping * 5
    min_model_no = 0
    t = large

    len_serial = len(serialize_validation_loss)
    # Find nearby Min candidates for recorded model weight
    for i in range(11):
        t_model = int(model_no + retrieve_mapping * 5 * (i - 5))
        if (t_model // retrieve_mapping) <= len_serial and (
        serialize_validation_loss[t_model // retrieve_mapping]) <= t:
            min_model_no = t_model
            t = (serialize_validation_loss[t_model // retrieve_mapping])
    candidates.append(min_model_no)

    # print(candidates)
    # print(serialize_validation_loss.index(min(serialize_validation_loss)))

min_model_no = max(set(candidates), key=candidates.count)

print('model.ckpt-' + str(min_model_no) + '.pt')

