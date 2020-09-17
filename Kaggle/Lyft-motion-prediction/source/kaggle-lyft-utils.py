
# image processing imports
import numpy as np

# machine learning imports

#  utiltites imports 
import pandas as pd
import cv2
import os
import random
from time import time


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)


def load_severstal_data(base_path='/home/isdgenomics/users/dipamcha/kaggle/severstal-steel-defect-detection/data/',
                        seed=0, masks_path=None, num_folds=5, fold=1):
    """ train_fault_only - Train on only faults, will train on only faulty images if true """
    train_df = pd.read_csv(base_path + 'train.csv')
    train_df['ImageId'] = train_df['ImageId_ClassId'].apply(
        lambda x: x.split('_')[0])
    train_df['ClassId'] = train_df['ImageId_ClassId'].apply(
        lambda x: x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

    if masks_path is not None:
        mask_df = pd.read_pickle(masks_path)
    else:
        masks_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
        masks_df.sort_values('hasMask', ascending=False, inplace=True)

    random_state = np.random.RandomState(seed=seed)
    all_idx = list(mask_df.index)
    random_state.shuffle(all_idx)

    assert fold <= num_folds and fold >= 1, "Fold number should be between 1 to num_folds"
    num_val = len(all_idx)//num_folds
    vs, ve = (fold-1)*num_val, fold*num_val
    val_idx = all_idx[vs:ve]
    train_idx = all_idx[:vs] + all_idx[ve:]

    return train_df, mask_df, train_idx, val_idx


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(256, 1600)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.bool)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def build_masks(rles, input_shape, background=True):
    depth = len(rles)
    if background:
        depth += 1
    height, width = input_shape
    masks = np.zeros((height, width, depth), np.bool)

    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, (width, height))

    if background:
        masks[:, :, -1] = np.logical_not(np.logical_or.reduce(masks, axis=-1))

    return masks


def build_rles(masks):
    width, height, depth = masks.shape

    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]

    return rles


# competition metric is mean over all individual masks
def dice_coef(y_true, y_pred, softmax_preds=True, thresh=0.5):
    _, h, w, c = y_pred.get_shape().as_list()
    if softmax_preds:
        y_pred_flat = tf.reshape(y_pred, [-1, h*w, c])
        y_pred_argmax = tf.math.argmax(y_pred_flat, axis=-1)
        y_pred_onehot = tf.one_hot(y_pred_argmax, c, on_value=1, off_value=0)
        y_pred_f = tf.cast(tf.reshape(
            y_pred_onehot, [-1, h*w, c]), tf.float32)[:, :, :-1]
        y_true_f = tf.reshape(y_true, [-1, h*w, c])[:, :, :-1]
    else:
        y_pred_f = tf.cast(tf.reshape(
            y_pred > thresh, [-1, h*w, c]), tf.float32)
        y_true_f = tf.reshape(y_true, [-1, h*w, c])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    masksum = tf.reduce_sum(y_true_f + y_pred_f, axis=1)
    dice = (2.*intersection + K.epsilon()) / (masksum + K.epsilon())
    return tf.reduce_mean(dice)


def gumbel_dice_loss(y_true, y_pred, temperature=1.):
    smooth = 0.1  # 1 #### 0.1 gives better dice score for noisy labels with this loss
    _, h, w, c = y_pred.get_shape().as_list()
    y_pred_f = tf.reshape(y_pred, [-1, h*w, c])
    y_true_f = tf.reshape(y_true, [-1, h*w, c])

    z = tf.random_uniform(tf.shape(y_pred_f))
    gumbel_z = tf.log(-tf.log(z))

    y_pred_tempscaled = y_pred_f / temperature
    y_pred_relaxedsample = tf.nn.softmax(y_pred_tempscaled - gumbel_z)
    intersection = tf.reduce_sum(y_true_f * y_pred_relaxedsample, axis=1)
    masksum = tf.reduce_sum(y_true_f + y_pred_relaxedsample, axis=1)
    dice = (2.*intersection + smooth) / (masksum + smooth)
    return 1 - tf.reduce_mean(dice)


def softmax_dice_loss(y_true, y_pred, alpha=1., gumbel_temp=0.1):
    if alpha > 0:
        loss = categorical_crossentropy(y_true, y_pred, from_logits=True) + \
            alpha*gumbel_dice_loss(y_true, y_pred, temperature=gumbel_temp)
        return loss
    else:
        return categorical_crossentropy(y_true, y_pred, from_logits=True)


def argmax_predictions(predictions):
    predflat = np.reshape(predictions, (-1, predictions.shape[-1]))
    p_am = np.argmax(predflat, axis=-1)
    outs = np.zeros(predflat.shape)
    outs[np.arange(p_am.size), p_am] = 1
    outs = np.reshape(outs, predictions.shape)
    return outs
