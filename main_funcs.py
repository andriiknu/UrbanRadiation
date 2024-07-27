from __future__ import print_function, division
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from sklearn.externals import joblib
from keras.layers import concatenate
from sklearn.preprocessing import RobustScaler
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, BatchNormalization, GlobalMaxPool1D
import os
from tqdm import tqdm
from collections import Counter
import gc
import warnings
warnings.filterwarnings("ignore")
try:
    import cPickle as pickle
except BaseException:
    import pickle
import multiprocessing


def train(expt_name, wdata_dir, random_seed=203):

    if random_seed: np.random.seed(random_seed)
    
    df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
    #=======================================================================================================================
    # Make weight directories
    weight_dir ='weights/' + expt_name + '/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    #=======================================================================================================================
    target = df_train.iloc[:, -1]
    y= to_categorical(target, num_classes=len(np.unique(target)))
    x_trn = df_train.iloc[:,1:-1]

    # scale train ==========================================================================================================
    X = x_trn.values
    where_are_NaNs = np.isnan(X)
    where_are_infs = np.isinf(X)
    X[where_are_NaNs] = 0
    X[where_are_infs] = 0

    scaler = RobustScaler()
    scaler.fit(X)
    # scaler_filename = "scaler.save"
    # joblib.dump(scaler, scaler_filename)

    scaled_train_X = scaler.transform(X)
    X = scaled_train_X
    X = X.reshape(len(df_train), len(X[0]), 1)

    #========================================================================================================================

    def init_model():
        inp = Input(shape=(len(X[0]), 1))

        a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(inp)
        a = BatchNormalization()(a)
        a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(a)
        a = BatchNormalization()(a)
        max_pool = GlobalMaxPool1D()(a)

        b = Flatten()(inp)
        ab = concatenate([ max_pool, b])

        a = Dense(128, activation="relu", kernel_initializer="uniform")(ab)
        a = Dropout(0.5)(a)
        a = Dense(128, activation="relu", kernel_initializer="uniform")(a)

        output = Dense(7, activation="softmax", kernel_initializer="uniform")(a)
        model = Model(inp, output)

        return model

    #======================================================================================================================
    num_folds = 5
    for i in range (7):
        _ids = df_train.index[df_train['152'] == i].tolist()
        all_length = len(_ids)
        fold_len = int(all_length / num_folds)

        init_idx = 0
        for j in range (num_folds):

            _train_idx = _ids[init_idx: init_idx + fold_len]
            init_idx = init_idx + fold_len
            df_train.loc[_train_idx, 'fold'] = j
    df_train = df_train.fillna(0)

    #=======================================================================================================================
    oof = np.zeros(shape = (len(df_train), 7))

    for fold_ in range(num_folds):
        trn_idx = df_train.index[df_train['fold'] != fold_].tolist()
        val_idx = df_train.index[df_train['fold'] == fold_].tolist()

        X_train, X_test = X[trn_idx], X[val_idx]
        y_train, y_test = y[trn_idx], y[val_idx]

        #===================================================================================================================
        callbacks = [EarlyStopping(monitor='val_acc',
                                patience=100,
                                verbose=2,
                                min_delta=1e-4,
                                mode='max'),
                    ReduceLROnPlateau(monitor='val_acc',
                                    factor=0.1,
                                    patience=50,
                                    cooldown=2,
                                    verbose=1,
                                    min_delta=1e-4,
                                    mode='max'),
                    ModelCheckpoint(monitor='val_acc',
                                    filepath=weight_dir + 'model_{}.hdf5'.format(fold_),
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='max'),
                    #TensorBoard(log_dir="logs/" + expt_name + '/'),
                    #SWA(weight_dir + 'model_swa_{}.hdf5'.format(fold_), 15)
                    ]

        # model training ===================================================================================================
        model = init_model()
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        # https://github.com/umbertogriffo/focal-loss-keras
        #model.compile(loss = [categorical_focal_loss(alpha=.25, gamma=0)], optimizer=Adam(lr=0.001), metrics=["accuracy"])

        epochs = 30
        model.fit(X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                verbose = 2,
                callbacks=callbacks)

        #===================================================================================================================
        model.load_weights(weight_dir + 'model_{}.hdf5'.format(fold_))
        pred_valid = model.predict(X_test)
        f1_err = f1_score(np.argmax(y_test, axis=1), np.argmax(pred_valid, axis=1), average='macro')
        print('F1 score on validation set:', f1_err)


    print('training complete.')

# function for finding source time by calculating mid segment of deected segments
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# # lstm model that runs on cpu
# def lstm_model():
#     inp = Input(shape=(99, 1))
#     a = LSTM(100, return_sequences=True, recurrent_activation='sigmoid')(inp)
#     a = LSTM(20, recurrent_activation='sigmoid')(a)
#     a = Dense(128, activation="relu", kernel_initializer="uniform")(a)
#     output = Dense(7, activation="softmax", kernel_initializer="uniform")(a)
#     model = Model(inp, output)
#     return model
#
# def ann_model():
#     inp = Input(shape=(len(X[0]), 1))
#
#     a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(inp)
#     a = BatchNormalization()(a)
#     a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(a)
#     a = BatchNormalization()(a)
#     max_pool = GlobalMaxPool1D()(a)
#
#     b = Flatten()(inp)
#     ab = concatenate([ max_pool, b])
#
#     a = Dense(128, activation="relu", kernel_initializer="uniform")(ab)
#     a = Dropout(0.5)(a)
#     a = Dense(128, activation="relu", kernel_initializer="uniform")(a)
#     output = Dense(7, activation="softmax", kernel_initializer="uniform")(a)
#     model = Model(inp, output)
#
#     return model
#=======================================================================================================================
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

def predict (test_folder, wdata_dir, seg_mul, const_seg_width=False, random_seed=203):


    np.random.seed(random_seed)

    files = sorted(os.listdir(test_folder))
    #=======================================================================================================================
    # folder name to save submits
    expt_name = 'ann'
    # weight directory
    expt_name1 = 'ANN_CNN'
    # expt_name2 = 'lstm'
    # expt_name3 = 'lgb'
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
    answer1_id = []
    answer1_fn = []
    answer1_tm = []
    answer2_id = []
    answer2_fn = []
    answer2_tm = []
    answer3_id = []
    answer3_fn = []
    answer3_tm = []
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

    model1 = load_model('weights/{}/model_{}.hdf5'.format(expt_name1, 0))
    model2 = load_model('weights/{}/model_{}.hdf5'.format(expt_name1, 1))
    model3 = load_model('weights/{}/model_{}.hdf5'.format(expt_name1, 2))
    model4 = load_model('weights/{}/model_{}.hdf5'.format(expt_name1, 3))
    model5 = load_model('weights/{}/model_{}.hdf5'.format(expt_name1, 4))

    print('model loaded')
    # model2 = lstm_model()
    # model2.load_weights('weights/{}/model_{}.hdf5'.format(expt_name2, fold_))
    # with open('weights/{}/model_{}.pkl'.format(expt_name3, fold_), 'rb') as fin:
    #     model3 = pickle.load(fin)


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
        #lgb
        scaled_train_X = scaler.transform(batch_inp_counts)
        # # lstm
        # X_lstm = scaled_train_X[:, 0:99]
        # X_lstm = X_lstm.reshape(len(seg_idxs), 99, 1)
        # ANN
        X = scaled_train_X.reshape(len(seg_idxs), len(X[0]), 1)
        # ========================================================
        # ANN
        pred1 = model1.predict(X, batch_size=len(seg_idxs))
        pred2 = model2.predict(X, batch_size=len(seg_idxs))
        pred3 = model3.predict(X, batch_size=len(seg_idxs))
        pred4 = model4.predict(X, batch_size=len(seg_idxs))
        pred5 = model5.predict(X, batch_size=len(seg_idxs))

        # # lstm
        # pred2 = model2.predict(X_lstm, batch_size=len(seg_idxs))
        # # lgb
        # pred3 = model3.predict(scaled_train_X)

        pred = pred1 + pred2 + pred3 + pred4 + pred5

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
        if freq_most_common > 7:
            # option 1: replace all non zero with most frequent
            # df['flag'] = np.where((df['label'] > 0), most_common, 0)
            # option 2: everything zero except most frequent
            df['flag'] = np.where((df['label'] != most_common), 0, most_common)

            preds_ = df['flag'].tolist()
            preds_nonzero = [x for x in preds_ if x > 0]
            most_common_, freq_most_common_ = Counter(preds_nonzero).most_common(1)[0]
            df['grad'] = smooth(df['flag'], freq_most_common_)
            nearest_time = df.loc[(df.grad.idxmax(), 'time_cumsum')]

            answer1_fn.append(id)
            answer1_id.append(most_common)
            answer1_tm.append(nearest_time / 1000000)
        else:
            answer1_fn.append(id)
            answer1_id.append(0)
            answer1_tm.append(0)

        if freq_most_common > 5:
            df['flag'] = np.where((df['label'] != most_common), 0, most_common)
            preds_ = df['flag'].tolist()
            preds_nonzero = [x for x in preds_ if x > 0]
            most_common_, freq_most_common_ = Counter(preds_nonzero).most_common(1)[0]
            df['grad'] = smooth(df['flag'], freq_most_common_)
            nearest_time = df.loc[(df.grad.idxmax(), 'time_cumsum')]

            answer2_fn.append(id)
            answer2_id.append(most_common)
            answer2_tm.append(nearest_time / 1000000)
        else:
            answer2_fn.append(id)
            answer2_id.append(0)
            answer2_tm.append(0)

        if freq_most_common > 3:
            df['flag'] = np.where((df['label'] != most_common), 0, most_common)
            preds_ = df['flag'].tolist()
            preds_nonzero = [x for x in preds_ if x > 0]
            most_common_, freq_most_common_ = Counter(preds_nonzero).most_common(1)[0]
            df['grad'] = smooth(df['flag'], freq_most_common_)
            nearest_time = df.loc[(df.grad.idxmax(), 'time_cumsum')]

            answer3_fn.append(id)
            answer3_id.append(most_common)
            answer3_tm.append(nearest_time / 1000000)
        else:
            answer3_fn.append(id)
            answer3_id.append(0)
            answer3_tm.append(0)

    # =======================================================================================================================

    sub1 = pd.DataFrame()
    sub1["RunID"] = answer1_fn
    sub1["SourceID"] = answer1_id
    sub1["SourceTime"] = answer1_tm
    sub1 = sub1.sort_values('RunID')
    sub1.to_csv(sub_dir + "solution_{}_th7_seg{}_test.csv".format(expt_name, seg_mul), index=False)

    sub2 = pd.DataFrame()
    sub2["RunID"] = answer2_fn
    sub2["SourceID"] = answer2_id
    sub2["SourceTime"] = answer2_tm
    sub2 = sub2.sort_values('RunID')
    sub2.to_csv(sub_dir + "solution_{}_th5_seg{}_test.csv".format(expt_name, seg_mul), index=False)

    sub3 = pd.DataFrame()
    sub3["RunID"] = answer3_fn
    sub3["SourceID"] = answer3_id
    sub3["SourceTime"] = answer3_tm
    sub3 = sub3.sort_values('RunID')
    sub3.to_csv(sub_dir + "solution_{}_th3_seg{}_test.csv".format(expt_name, seg_mul), index=False)

    print('done')
    
def predict_3tta(test_folder, wdata_dir):
    p1 = multiprocessing.Process(target=predict, args=(test_folder, wdata_dir, 1.25, False))
    p2 = multiprocessing.Process(target=predict, args=(test_folder, wdata_dir, 1500, True))
    p3 = multiprocessing.Process(target=predict, args=(test_folder, wdata_dir, 3000, True))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    print("All predictions completed.")

def vote_ensemble(wdata_dir):

    # folder name to save submits
    expt_name = 'ann'
    sub_dir = wdata_dir + 'submits/' + expt_name + '/'
    threshold = 4  # oof best 4


    # df1 = pd.read_csv(sub_dir + 'solution_{}_th3_seg0.5_test.csv'.format(expt_name))
    # df2 = pd.read_csv(sub_dir + 'solution_{}_th5_seg0.5_test.csv'.format(expt_name))
    # df3 = pd.read_csv(sub_dir + 'solution_{}_th7_seg0.5_test.csv'.format(expt_name))

    df4 = pd.read_csv(sub_dir + 'solution_{}_th3_seg1.25_test.csv'.format(expt_name))
    df5 = pd.read_csv(sub_dir + 'solution_{}_th5_seg1.25_test.csv'.format(expt_name))
    df6 = pd.read_csv(sub_dir + 'solution_{}_th7_seg1.25_test.csv'.format(expt_name))

    # df7 = pd.read_csv(sub_dir + 'solution_{}_th3_seg0.33_test.csv'.format(expt_name))
    # df8 = pd.read_csv(sub_dir + 'solution_{}_th5_seg0.33_test.csv'.format(expt_name))
    # df9 = pd.read_csv(sub_dir + 'solution_{}_th7_seg0.33_test.csv'.format(expt_name))

    df10 = pd.read_csv(sub_dir + 'solution_{}_th3_seg1500_test.csv'.format(expt_name))
    df11 = pd.read_csv(sub_dir + 'solution_{}_th5_seg1500_test.csv'.format(expt_name))
    df12 = pd.read_csv(sub_dir + 'solution_{}_th7_seg1500_test.csv'.format(expt_name))

    df13 = pd.read_csv(sub_dir + 'solution_{}_th3_seg3000_test.csv'.format(expt_name))
    df14 = pd.read_csv(sub_dir + 'solution_{}_th5_seg3000_test.csv'.format(expt_name))
    df15 = pd.read_csv(sub_dir + 'solution_{}_th7_seg3000_test.csv'.format(expt_name))

    ids = df4.RunID
    sources = df4.SourceID

    df_combined = pd.DataFrame()
    df_combined['RunID'] = ids

    # sid1 = df1.SourceID.values
    # sid2 = df2.SourceID.values
    # sid3 = df3.SourceID.values
    sid4 = df4.SourceID.values
    sid5 = df5.SourceID.values
    sid6 = df6.SourceID.values
    # sid7 = df7.SourceID.values
    # sid8 = df8.SourceID.values
    # sid9 = df9.SourceID.values
    sid10 = df10.SourceID.values
    sid11 = df11.SourceID.values
    sid12 = df12.SourceID.values
    sid13 = df13.SourceID.values
    sid14 = df14.SourceID.values
    sid15 = df15.SourceID.values

    # time1 = df1.SourceTime.values
    # time2 = df2.SourceTime.values
    # time3 = df3.SourceTime.values
    time4 = df4.SourceTime.values
    time5 = df5.SourceTime.values
    time6 = df6.SourceTime.values
    # time7 = df7.SourceTime.values
    # time8 = df8.SourceTime.values
    # time9 = df9.SourceTime.values
    time10 = df10.SourceTime.values
    time11 = df11.SourceTime.values
    time12 = df12.SourceTime.values
    time13 = df13.SourceTime.values
    time14 = df14.SourceTime.values
    time15 = df15.SourceTime.values

    np_sid = np.array([#sid1, sid2, sid3,
                    sid4, sid5, sid6,
                    #sid7, sid8, sid9,
                    sid10, sid11, sid12,
                    sid13, sid14, sid15
                    ]).T
    np_time = np.array([#time1, time2, time3,
                        time4, time5, time6,
                        #time7, time8, time9,
                        time10, time11, time12,
                        time13, time14, time15
                        , ], dtype=np.float16).T

    run_id = []
    filtered_label = []
    filtered_time = []

    for i, rid in enumerate(ids):
        all_labels = np_sid[i]
        all_timess = np_time[i]

        # Count the frequency of non-zero preds
        try:
            preds_nonzero = [x for x in all_labels if x > 0]
            most_common, freq_most_common = Counter(preds_nonzero).most_common(1)[0]
        except:
            freq_most_common = 0

        #most_common, freq_most_common = Counter(all_labels).most_common(1)[0]

        if freq_most_common >= threshold:
            idx = np.where(all_labels==most_common)
            _time = all_timess[idx]
            avg_time = sum(_time)/len(_time)

            run_id.append(rid)
            filtered_label.append(most_common)
            filtered_time.append(avg_time)
            #print('done')

        else:
            run_id.append(rid)
            filtered_label.append(0)
            filtered_time.append(0)

    sub = pd.DataFrame()
    sub["RunID"] = run_id
    sub['SourceID'] = filtered_label
    sub["SourceTime"] = filtered_time

    print(sub['SourceID'].astype(bool).sum(axis=0))

    sub.to_csv(sub_dir + "{}_3tta_th{}_test.csv".format(expt_name, threshold), index=False)


    print('done')

def time_process(test_folder, wdata_dir, solution_fn):

    def find_counts(en, src_id):

        # convert energy values to bin counts and normalize
        out = pd.cut(en, bins=bins, include_lowest=True)
        counts = out.value_counts(sort=False)
        np_counts = np.array(counts.values, dtype=np.float32)
        np_counts = np_counts / np.sum(np_counts)


        np_counts = np_counts.reshape(1, 99)
        np_counts = scaler.transform(np_counts)
        np_counts = np.squeeze(np_counts)

        if src_id == 1: # HEU
            # HEU
            count1 = np_counts[0]
            count2 = np_counts[3]
            count3 = np_counts[6]
            count4 = np_counts[17]
            count5 = np_counts[19]
            count6 = np_counts[87]

            total_counts = count1 + count2 + count3 + count4 + count5 + count6
            #total_counts = count4 + count5 + count6

        elif src_id == 2: # WPu 77, 407, 653 // 33, 61, 101, 207, 381
            count1 = np_counts[1]
            count2 = np_counts[2]
            count3 = np_counts[3]
            count4 = np_counts[6]
            count5 = np_counts[12]
            count6 = np_counts[13]
            count7 = np_counts[21]

            total_counts = count1 + count2 + count3 + count4 + count5 + count6 + count7
            #total_counts = count5 + count6 + count7

        elif src_id == 3: # I
            count1 = np_counts[1]
            count2 = np_counts[2]
            count3 = np_counts[6]
            count4 = np_counts[9]
            count5 = np_counts[12]
            count6 = np_counts[21]
            total_counts = count1 + count2 + count3 + count4 + count5 + count6
            #total_counts = count4 + count5 + count6

        elif src_id == 4: # Co
            count1 = np_counts[39]
            count2 = np_counts[43]
            count3 = np_counts[44]
            total_counts = count1 + count2 + count3

        elif src_id == 5: # Tc
            count1 = np_counts[0]
            count2 = np_counts[1]
            count3 = np_counts[4]
            count4 = np_counts[10]
            total_counts = count1 + count2 + count3 + count4

        elif src_id == 6: # HEU+Tc
            count1 = np_counts[0]
            count2 = np_counts[3]
            count3 = np_counts[6]
            count4 = np_counts[17]
            count5 = np_counts[19]
            count6 = np_counts[87]

            count7 = np_counts[1]
            count8 = np_counts[4]
            count9 = np_counts[10]

            total_counts = count1 + count2 + count3 + count4 + count5 + count6 + count7 + count8 + count9
        else:
            total_counts = 0

        return total_counts

    # A function that can be called to do work:
    def work(arg):
        # Split the list to individual variables:
        i, id = arg
        id = int(id)
        source_id = source_ids[i]
        apprx_time = coarse_time[i]

        if source_id == 0:
            return id,source_id, 0

        else:
            #df = pd.read_csv(data_dir + 'testing/{}.csv'.format(id))
            df = pd.read_csv(test_folder + '{}.csv'.format(id))
            time = df[df.columns[0]]
            energy = df[df.columns[1]]
            df['time_cumsum'] = np.array(time.cumsum(), dtype=np.float32)
            length = len(time)

            #=====================================================================
            # divide test file into 30 segments to get unit segment size
            seg_width = int(length / 30)
            if seg_width > 20000: seg_width = 20000
            if seg_width < 1000: seg_width = 1000
            # find 30 sec index
            df_sort = df.loc[(df['time_cumsum'] - 30000000).abs().argsort()[:2]]
            idx30 = df_sort.index.tolist()[0]

            # find apprx time index
            df_sort_apprx = df.loc[(df['time_cumsum'] - apprx_time*1000000).abs().argsort()[:2]]
            idx_apprx = df_sort_apprx.index.tolist()[0]
            #===================================================================

            #----------------------------------------------------------------------------------
            #                       center of scan window
            #                                  /
            #---------------|--------|---------/----------|----------|--------------------------
            #               |        |         /          |          |
            # --------------|--------|---------/----------|----------|--------------------------
            #                                  /
            #========================|  1000 data points  |=====================================
            #===============|           1 segment data points        |==========================
            #-----------------------------------------------------------------------------------

            # average time from different segment sizes (range 0f scan)
            #seg_mul = [0.25, 0.5, 0.75]             # 93.14 //
            seg_mul = [0.75]
            #seg_mul = [1.5]
            preds = 0
            for k in seg_mul:  # search range
                start = idx_apprx - seg_width*k
                end = idx_apprx + seg_width*k
                if start < idx30: start = idx30
                if end > length-seg_width*k: end = length-seg_width*k

                for j in range (int(start), int(end), propagation_step):
                    # # resolution 1
                    en1 = energy[j - 500 : j + 500]
                    total_cnts1 = find_counts(en1, source_id)

                    # resolution 2
                    en2 = energy[j - int(seg_width * 0.25) : j + int(seg_width * 0.25)]  # window for max count
                    total_cnts2 = find_counts(en2, source_id)

                    # resolution 3
                    en3 = energy[j - int(seg_width * 0.5) : j + int(seg_width * 0.5)]  # window for max count
                    total_cnts3 = find_counts(en3, source_id)

                    # resolution 4
                    en4 = energy[j - int(seg_width * 0.66) : j + int(seg_width * 0.66)]  # window for max count
                    total_cnts4 = find_counts(en4, source_id)

                    df.at[j, 'peak_counts_{}'.format(k)] = total_cnts1 + total_cnts2 + total_cnts3 + total_cnts4
                    #df.at[j, 'peak_counts_{}'.format(k)] = total_cnts3

                _df = df[np.isfinite(df['peak_counts_{}'.format(k)])]
                time_at_max_count = _df.loc[_df['peak_counts_{}'.format(k)].idxmax(), 'time_cumsum']
                pred_t = time_at_max_count/1000000
                preds = preds + pred_t

                del _df
                gc.collect()

            pred = preds/len(seg_mul)

            return id, source_id, pred

    # folder name to save submits
    expt_name = 'ann'
    sub_dir = wdata_dir + 'submits/' + expt_name + '/'
    df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
    #######################################################################################################
    # submit without pseudo
    input_fn = 'ann_3tta_th4_test.csv'
    #######################################################################################################
    input_df = pd.read_csv(sub_dir + input_fn )
    propagation_step = 100

    test_ids = input_df.RunID
    source_ids = input_df.SourceID
    coarse_time = input_df.SourceTime
    #=======================================================================================================================

    x_trn = df_train.iloc[:,1:100]
    # scale train
    X = x_trn.values
    where_are_NaNs = np.isnan(X)
    where_are_infs = np.isinf(X)
    X[where_are_NaNs] = 0
    X[where_are_infs] = 0


    scaler = RobustScaler()
    scaler.fit(X)
    scaled_train_X = scaler.transform(X)
    X = scaled_train_X

    #scaler = joblib.load("scaler.save")
    # bins for test segment
    bins = np.arange(0,3000,30)

    #=======================================================================================================================
    # Parallel code
    idx_list = list(input_df.index.values)
    id_list = test_ids.tolist()
    arg_instances =  list(zip(idx_list, id_list))
    # parallel processing
    results = Parallel(n_jobs=8*2, verbose=50, batch_size=2)(map(delayed(work), arg_instances))
    #=======================================================================================================================
    # write submission file
    df_pred_time = pd.DataFrame(results)
    df_pred_time.columns = ["RunID", "SourceID", "SourceTime"]
    df_pred_time.to_csv(sub_dir + 'solution_{}_mul75_robust_peakall.csv'.format(input_fn), index = False)
    df_pred_time.to_csv(solution_fn, index = False)

    print('All done. Final predictions are saved in current directory.')