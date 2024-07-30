import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from keras.models import load_model
from tqdm import tqdm
from collections import Counter
import os

def make_features(energy, energy_bins):
    out = pd.cut(energy, bins=energy_bins, include_lowest=True)
    counts = out.value_counts(sort=False)
    np_counts = np.array(counts.values, dtype=np.float32)
    np_counts = np_counts / np.sum(np_counts)

    # ===================================================================================================================
    # peak to compton ratio feature
    # ===================================================================================================================

    count1 = np_counts[1] / np.sum(np_counts[0:1])
    count2 = np_counts[2] / np.sum(np_counts[0:2])
    count3 = np_counts[3] / np.sum(np_counts[0:2])
    count4 = np_counts[4] / np.sum(np_counts[0:3])
    count5 = np_counts[5] / np.sum(np_counts[0:4])
    count6 = np_counts[6] / np.sum(np_counts[0:5])
    count7 = np_counts[7] / np.sum(np_counts[0:6])
    count8 = np_counts[8] / np.sum(np_counts[0:7])
    count9 = np_counts[9] / np.sum(np_counts[0:8])
    count10 = np_counts[10] / np.sum(np_counts[0:9])
    count11 = np_counts[11] / np.sum(np_counts[0:10])
    count12 = np_counts[12] / np.sum(np_counts[0:11])
    count13 = np_counts[13] / np.sum(np_counts[0:12])
    count14 = np_counts[14] / np.sum(np_counts[0:13])
    count15 = np_counts[15] / np.sum(np_counts[0:14])
    count16 = np_counts[16] / np.sum(np_counts[0:15])
    count17 = np_counts[17] / np.sum(np_counts[0:16])
    count18 = np_counts[18] / np.sum(np_counts[0:17])

    count19 = np_counts[19] / np.sum(np_counts[0:18])
    count20 = np_counts[20] / np.sum(np_counts[0:19])
    count21 = np_counts[21] / np.sum(np_counts[0:20])
    count22 = np_counts[87] / np.sum(np_counts[0:86])

    np_counts_peaks = np.array(
        [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10,
         count11, count12, count13, count14, count15, count16, count17, count18, count19, count20,
         count21, count22]).T

    # ===================================================================================================================
    # peak to peak ratio feature
    # ===================================================================================================================
    # HEU
    ratio1 = np_counts[0] / (np_counts[17] + np_counts[19])
    ratio2 = np_counts[3] / (np_counts[17] + np_counts[19])
    ratio3 = np_counts[6] / (np_counts[17] + np_counts[19])
    ratio4 = np_counts[87] / (np_counts[17] + np_counts[19])

    # WGPu
    ratio5 = np_counts[1] / (np_counts[12] + np_counts[13])
    ratio6 = np_counts[2] / (np_counts[12] + np_counts[13])
    ratio7 = np_counts[3] / (np_counts[12] + np_counts[13])
    ratio8 = np_counts[6] / (np_counts[12] + np_counts[13])
    ratio9 = np_counts[21] / (np_counts[12] + np_counts[13])

    # I-131
    ratio10 = np_counts[1] / np_counts[12]
    ratio11 = np_counts[2] / np_counts[12]
    ratio12 = np_counts[6] / np_counts[12]
    ratio13 = np_counts[9] / np_counts[12]
    ratio14 = np_counts[21] / np_counts[12]

    # Tc-99m
    ratio15 = np_counts[0] / np_counts[4]
    ratio16 = np_counts[1] / np_counts[4]
    ratio17 = np_counts[10] / np_counts[4]

    # HEU + Tc-99m
    ratio18 = np.sum(np_counts[0:7]) / np.sum(np_counts[0:21])
    ratio19 = ratio1 / ratio15
    ratio20 = ratio2 / ratio15
    ratio21 = ratio3 / ratio15
    ratio22 = ratio4 / ratio15

    ratio23 = ratio1 / ratio16
    ratio24 = ratio2 / ratio16
    ratio25 = ratio3 / ratio16
    ratio26 = ratio4 / ratio16

    ratio27 = ratio1 / ratio17
    ratio28 = ratio2 / ratio17
    ratio29 = ratio3 / ratio17
    ratio30 = ratio4 / ratio17

    np_ratio_peaks = np.array(
        [ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8, ratio9, ratio10,
         ratio11, ratio12, ratio13, ratio14, ratio15, ratio16, ratio17, ratio18, ratio19, ratio20,
         ratio21, ratio22, ratio23, ratio24, ratio25, ratio26, ratio27, ratio28, ratio29, ratio30]).T


    feats  = np.concatenate([np_counts, np_counts_peaks, np_ratio_peaks], axis=0)
    return feats

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def process_threshold(df, id, most_common, freq_most_common, threshold, answer_fn, answer_id, answer_tm):
    if freq_most_common > threshold:
        df['flag'] = np.where((df['label'] != most_common), 0, most_common)
        preds_ = df['flag'].tolist()
        preds_nonzero = [x for x in preds_ if x > 0]
        most_common_, freq_most_common_ = Counter(preds_nonzero).most_common(1)[0]
        df['grad'] = smooth(df['flag'], freq_most_common_)
        nearest_time = df.loc[(df.grad.idxmax(), 'time_cumsum')]

        answer_fn.append(id)
        answer_id.append(most_common)
        answer_tm.append(nearest_time / 1000000)
    else:
        answer_fn.append(id)
        answer_id.append(0)
        answer_tm.append(0)

def predict (test_folder, wdata_dir, seg_mul, threshold=4, const_seg_width=False, random_seed=203):

    np.random.seed(random_seed)

    files = sorted(os.listdir(test_folder))
    #=======================================================================================================================
    # folder name to save submits
    expt_name = 'ann'
    # weight directory
    expt_name1 = 'ANN_CNN'
    # #=======================================================================================================================
    # read data
    #=======================================================================================================================
    #df_test = pd.read_csv(data_dir + 'submittedAnswers.csv')
    df_feat = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
    # =======================================================================================================================

    #======================================================================================================================
    # make bins like train
    energy_bin_size = 30
    energy_bins = np.arange(0, 3000, energy_bin_size)
    num_windows = 200

    # Make submission directories
    sub_dir = wdata_dir + 'submits/' + expt_name + '/'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # =======================================================================================================================
    # =======================================================================================================================
    # Empty list list tostore results
    answer_id = []
    answer_fn = []
    answer_tm = []

    # =======================================================================================================================
    # stats from train set
    # =======================================================================================================================
    #target = df_train.iloc[:, -1]
    x_trn = df_feat.iloc[:,1:-1]
    # scale parameters from train
    X = x_trn.values

    where_are_NaNs = np.isnan(X)
    where_are_infs = np.isinf(X)
    X[where_are_NaNs] = 0
    X[where_are_infs] = 0
    scaler = RobustScaler()
    scaler.fit(X)
    # =======================================================================================================================
    # iterate over each file from test set
    # =======================================================================================================================

    model = load_model('weights/{}/model.hdf5'.format(expt_name1))

    print('model loaded')

    for i, id in enumerate(tqdm(files)):

        id = os.path.splitext(id)[0]
        df = pd.read_csv(test_folder + '{}.csv'.format(id))

        time = df[df.columns[0]]
        energy = df[df.columns[1]]
        length = len(time)

        df['time_cumsum'] = np.array(time.cumsum(), dtype=np.float32)


        # divide test file into 30 segments
        if not const_seg_width: 
            seg_width = int(length / 30)
            if seg_width > 10000: seg_width = 10000
            if seg_width < 2000: seg_width = 2000
        else: seg_width = 1      


        # Create equally spaced slices
        f = lambda m, n: [k * n // m + n // (2 * m) for k in range(m)]
        _idxs = f(num_windows, length)  # 200 slices

        # find 30 sec index
        df_sort = df.loc[(df['time_cumsum'] - 30000000).abs().argsort()[:2]]
        idx30 = df_sort.index.tolist()[0]
        seg_idxs = [k for k in _idxs if k >= idx30]

        batch_inp_counts = np.zeros((len(seg_idxs), len(X[0]), 1))
        for s, j in enumerate(seg_idxs):

            ###################################################
            start = int(j - seg_width * seg_mul)
            end = int(j + seg_width * seg_mul)
            if j - seg_width * seg_mul < 0: start = 0
            if j + seg_width * seg_mul > length: end = length
            ###################################################

            seg = df[start:end]
            energy = seg[seg.columns[1]]
            features = make_features(energy, energy_bins)

            where_are_NaNs = np.isnan(features)
            where_are_infs = np.isinf(features)
            features[where_are_NaNs] = 0
            features[where_are_infs] = 0

            ##################################################
            # df_counts = np_counts.reshape(1,99,1)
            inp_counts = features
            inp_counts = inp_counts.reshape(len(features), 1)
            batch_inp_counts[s, :, ] = inp_counts
            ##################################################

        batch_inp_counts = np.squeeze(batch_inp_counts)
        # ==========================================================
        # scale input
 
        scaled_train_X = scaler.transform(batch_inp_counts)

        # ANN
        X = scaled_train_X.reshape(len(seg_idxs), len(X[0]), 1)
        pred = model.predict(X, batch_size=len(seg_idxs))
        p = pred.argmax(axis=1)

        # update dataframe
        for m, n in enumerate(seg_idxs):
            df.loc[int(n), 'label'] = p[m]

        df = df[np.isfinite(df['label'])]
        # ===================================================================================================================
        # Count the frequency of non-zero preds
        try:
            preds_nonzero = [x for x in p if x > 0]
            most_common, freq_most_common = Counter(preds_nonzero).most_common(1)[0]
        except:
            freq_most_common = 0

        # ===================================================================================================================
        # If frequency of non-zero predictions are greater than a threshold, then it's positive
        process_threshold(df, id, most_common, freq_most_common, threshold, answer_fn, answer_id, answer_tm)

    # =======================================================================================================================

    sub = pd.DataFrame()
    sub["RunID"] = answer_fn
    sub["SourceID"] = answer_id
    sub["SourceTime"] = answer_tm
    sub = sub.sort_values('RunID')
    sub.to_csv(sub_dir + "solution_{}_th{}_seg{}_test.csv".format(expt_name, seg_mul), index=False)

    print('done')