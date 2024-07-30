import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, BatchNormalization, GlobalMaxPool1D, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score
import os

def get_training_data(df_train):
    target = df_train.iloc[:, -1]
    y = to_categorical(target, num_classes=len(np.unique(target)))
    x_trn = df_train.iloc[:, 1:-1]  

    # Scale train data
    X = x_trn.values
    where_are_NaNs = np.isnan(X)
    where_are_infs = np.isinf(X)
    X[where_are_NaNs] = 0
    X[where_are_infs] = 0

    scaler = RobustScaler()
    scaler.fit(X)
    scaled_train_X = scaler.transform(X)
    X = scaled_train_X
    X = X.reshape(len(df_train), len(X[0]), 1)

    return (X, y)

def fold_data(df_train, num_folds=5, num_classes=7):
    for i in range(num_classes):
        _ids = df_train.index[df_train['152'] == i].tolist()
        all_length = len(_ids)
        fold_len = int(all_length / num_folds)

        init_idx = 0
        for j in range(num_folds):
            _train_idx = _ids[init_idx: init_idx + fold_len]
            init_idx = init_idx + fold_len
            df_train.loc[_train_idx, 'fold'] = j
    df_train = df_train.fillna(0)

def init_model(num_features):
    inp = Input(shape=(num_features, 1))

    a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform")(inp)
    a = BatchNormalization()(a)
    a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform")(a)
    a = BatchNormalization()(a)
    max_pool = GlobalMaxPool1D()(a)

    b = Flatten()(inp)
    ab = concatenate([max_pool, b])

    a = Dense(128, activation="relu", kernel_initializer="uniform")(ab)
    a = Dropout(0.5)(a)
    a = Dense(128, activation="relu", kernel_initializer="uniform")(a)

    output = Dense(7, activation="softmax", kernel_initializer="uniform")(a)
    model = Model(inp, output)

    return model

def train(expt_name, wdata_dir, random_seed=203):
    if random_seed: 
        np.random.seed(random_seed)
    
    df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
    
    # Make weight directories
    weight_dir = 'weights/' + expt_name + '/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # Get training data
    X, y = get_training_data(df_train)

    # Callbacks
    log_dir = "logs/" + expt_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    callbacks = [
        EarlyStopping(monitor='acc', patience=20, verbose=2, min_delta=1e-4, mode='max'),
        ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5, cooldown=2, verbose=1, min_delta=1e-4, mode='max'),
        ModelCheckpoint(monitor='acc', filepath=weight_dir + 'model.hdf5', save_best_only=True, save_weights_only=False, mode='max'  ),
        TensorBoard(log_dir=log_dir)
    ]
    
    # Model training
    num_features = X.shape[1]
    model = init_model(num_features)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

    history = model.fit(X, y,
                        epochs=30,
                        batch_size=256,
                        shuffle=True,
                        verbose=2,
                        callbacks=callbacks)

    return model, history

def train_folds(expt_name, wdata_dir, random_seed=203):
    if random_seed: 
        np.random.seed(random_seed)
    
    df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
    
    # Make weight directories
    weight_dir = 'weights/' + expt_name + '/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # Get training data
    X, y = get_training_data(df_train)

    # Fold data
    num_folds = 5
    fold_data(df_train, num_folds=num_folds)

    for fold_ in range(num_folds):
        trn_idx = df_train.index[df_train['fold'] != fold_].tolist()
        val_idx = df_train.index[df_train['fold'] == fold_].tolist()

        X_train, X_test = X[trn_idx], X[val_idx]
        y_train, y_test = y[trn_idx], y[val_idx]

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=100, verbose=2, min_delta=1e-4, mode='max'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=50, cooldown=2, verbose=1, min_delta=1e-4, mode='max'),
            ModelCheckpoint(monitor='val_accuracy', filepath=weight_dir + 'model_{}.hdf5'.format(fold_), save_best_only=True, save_weights_only=False, mode='max')
        ]

        # Model training
        num_features = X.shape[1]
        model = init_model(num_features)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        epochs = 30
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=epochs,
                  batch_size=128,
                  shuffle=True,
                  verbose=2,
                  callbacks=callbacks)

        model.load_weights(weight_dir + 'model_{}.hdf5'.format(fold_))
        pred_valid = model.predict(X_test)
        f1_err = f1_score(np.argmax(y_test, axis=1), np.argmax(pred_valid, axis=1), average='macro')
        print('F1 score on validation set:', f1_err)

    print('Training complete.')