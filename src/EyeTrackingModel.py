import os
import sys
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

lr = 1e-3
epochs = 50
batch_size = 128
SEED = 64
tf.random.set_seed(SEED)
np.random.seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(8)


class LogisticModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = "LogisticModel"
        self.dir_path = dir_path
        self.user_session_cols = ['MMF_y_inf', 'MMF_x_inf', 'MSF_y_inf', 'MSF_clk']
        self.user_tmp_cols = ['isVisible', "mouseX", "mouseY",
                              'M_tclk', 'S_cy', 'S_h']
        self.msg_tmp_cols = ['MSG_y', "MSG_h", "MSG_tclk"]
        self.msg_session_cols = ['time_1', 'time_2', 'time_3', 'MSG_tmy', 'MSG_sh', 'MSG_sy', 'MSG_clk', 'MSG_svt']
        self.info_cols = ['userId', 'postId', "newsletter", "title", 'timestamp', "index", "read"]

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        X_train1 = X_train[self.user_tmp_cols + self.msg_tmp_cols].values
        X_val1 = X_val[self.user_tmp_cols + self.msg_tmp_cols].values
        y_train = y_train.values
        y_val = y_val.values
        user_msg_input = Input(shape=(len(self.user_tmp_cols) + len(self.msg_tmp_cols),))
        z = Dense(1)(user_msg_input)
        model = Model(inputs=[user_msg_input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
        print("[INFO] training model...")
        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32')], y=y_train,
            sample_weight=np.asarray([20.0 if y > 0 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        model.save(self.save_path + self.name + ".h5")
        self.model = model

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict([df[self.user_tmp_cols + self.msg_tmp_cols].values.astype('float32')])
        preds = np.exp(preds)
        X_test["pred"] = preds
        X_test["target"] = y_test
        pred_s = X_test.groupby(by=["userId", "postId", "timestamp"])["pred"].sum().reset_index()
        X_test = pd.merge(X_test, pred_s, on=["userId", "postId", "timestamp"])
        X_test["pred"] = X_test["pred_x"] / X_test["pred_y"]

        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class NNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = "NNModel"
        self.dir_path = dir_path
        self.user_session_cols = ['MMF_y_inf', 'MMF_x_inf', 'MSF_y_inf', 'MSF_clk']
        self.user_tmp_cols = ['isVisible', "mouseX", "mouseY",
                              'M_tclk', 'S_cy', 'S_h']
        self.msg_tmp_cols = ['MSG_y', "MSG_h", "MSG_tclk"]
        self.msg_session_cols = ['time_1', 'time_2', 'time_3', 'MSG_tmy', 'MSG_sh', 'MSG_sy', 'MSG_clk', 'MSG_svt']
        self.info_cols = ['userId', 'postId',  "newsletter", "title", 'timestamp', "index", "read"]

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        X_train1 = X_train[self.user_tmp_cols + self.msg_tmp_cols].values
        X_val1 = X_val[self.user_tmp_cols + self.msg_tmp_cols].values
        y_train = y_train.values
        y_val = y_val.values
        user_msg_input = Input(shape=(len(self.user_tmp_cols) + len(self.msg_tmp_cols),))

        x = Dense(16, activation="relu")(user_msg_input)
        x = Dense(8, activation="relu")(x)
        z = Dense(1)(x)
        model = Model(inputs=[user_msg_input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
        print("[INFO] training model...")

        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32')], y=y_train,
            sample_weight=np.asarray([20.0 if y > 0 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        model.save(self.save_path + self.name + ".h5")
        self.model = model

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict([df[self.user_tmp_cols + self.msg_tmp_cols].values.astype('float32')])
        preds = np.exp(preds)
        X_test["pred"] = preds
        X_test["target"] = y_test
        pred_s = X_test.groupby(by=["userId", "postId", "timestamp"])["pred"].sum().reset_index()
        X_test = pd.merge(X_test, pred_s, on=["userId", "postId", "timestamp"])
        X_test["pred"] = X_test["pred_x"] / X_test["pred_y"]

        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class PatternNNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = "PatternNNModel"
        self.dir_path = dir_path
        self.user_pattern_cols = ['MMF_y_2', 'MMF_y_5', 'MMF_y_10', 'MMF_y_inf', 'MMF_x_2', 'MMF_x_5', 'MMF_x_10',
                                  'MMF_x_inf',
                                  'MSF_y_2', 'MSF_y_5', 'MSF_y_10', 'MSF_y_inf', 'MSF_clk']
        self.user_session_cols = ['MMF_y_inf', 'MMF_x_inf', 'MSF_y_inf', 'MSF_clk']
        self.user_tmp_cols = ['isVisible', "mouseX", "mouseY",
                              'M_tclk', 'S_cy', 'S_h']
        self.msg_tmp_cols = ['MSG_y', "MSG_h", "MSG_tclk"]
        self.msg_session_cols = ['time_1', 'time_2', 'time_3', 'MSG_tmy', 'MSG_sh', 'MSG_sy', 'MSG_clk', 'MSG_svt']
        self.info_cols = ['userId', 'postId',  "newsletter", "title", 'timestamp', "index", "read"]

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        X_train1 = X_train[self.user_tmp_cols + self.msg_tmp_cols].values
        X_train2 = X_train[self.user_pattern_cols].values
        X_val1 = X_val[self.user_tmp_cols + self.msg_tmp_cols].values
        X_val2 = X_val[self.user_pattern_cols].values
        y_train = y_train.values
        user_pattern_input = Input(shape=(len(self.user_pattern_cols),))
        user_msg_input = Input(shape=(len(self.user_tmp_cols) + len(self.msg_tmp_cols),))

        x = Dense(16, activation="relu")(user_msg_input)
        x = Dense(8, activation="relu")(x)
        x = Model(inputs=user_msg_input, outputs=x)
        y = Dense(16, activation="relu")(user_pattern_input)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=user_pattern_input, outputs=y)
        combined = tf.keras.layers.Multiply()([x.output, y.output])
        z = Dense(1)(combined)
        model = Model(inputs=[x.input, y.input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
        print("[INFO] training model...")
        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32'), X_train2.astype('float32')], y=y_train,
            sample_weight=np.asarray([20.0 if y > 0 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32'), X_val2.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        model.save(self.save_path + self.name + ".h5")
        self.model = model

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict([df[self.user_tmp_cols + self.msg_tmp_cols].values.astype('float32'),
                                    df[self.user_pattern_cols].values.astype('float32')])
        preds = np.exp(preds)
        X_test["pred"] = preds
        X_test["target"] = y_test
        pred_s = X_test.groupby(by=["userId", "postId", "timestamp"])["pred"].sum().reset_index()
        X_test = pd.merge(X_test, pred_s, on=["userId", "postId", "timestamp"])
        X_test["pred"] = X_test["pred_x"] / X_test["pred_y"]

        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class BaselineNNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = "BaselineNNModel"
        self.dir_path = dir_path
        self.msg_tmp_cols = ['height_1', "height_2", "height_3", "MSG_tclk"]
        self.info_cols = ['userId', 'postId',"newsletter", "title", 'timestamp', "index", "read"]

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        X_train1 = X_train[self.msg_tmp_cols].values
        X_val1 = X_val[self.msg_tmp_cols].values
        y_train = y_train.values
        y_val = y_val.values
        user_msg_input = Input(shape=(len(self.msg_tmp_cols),))

        x = Dense(16, activation="relu")(user_msg_input)
        x = Dense(8, activation="relu")(x)
        z = Dense(1)(x)
        model = Model(inputs=[user_msg_input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
        print("[INFO] training model...")
        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32')], y=y_train,
            sample_weight=np.asarray([20.0 if y > 0 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        model.save(self.save_path + self.name + ".h5")
        self.model = model

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict([df[self.msg_tmp_cols].values.astype('float32')])
        preds = np.exp(preds)
        X_test["pred"] = preds
        X_test["target"] = y_test
        pred_s = X_test.groupby(by=["userId", "postId", "timestamp"])["pred"].sum().reset_index()
        X_test = pd.merge(X_test, pred_s, on=["userId", "postId", "timestamp"])
        X_test["pred"] = X_test["pred_x"] / X_test["pred_y"]

        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class PatternBaselineNNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = "PatternBaselineNNModel"
        self.dir_path = dir_path
        self.user_pattern_cols = ['MMF_y_2', 'MMF_y_5', 'MMF_y_10', 'MMF_y_inf', 'MMF_x_2', 'MMF_x_5', 'MMF_x_10',
                                  'MMF_x_inf',
                                  'MSF_y_2', 'MSF_y_5', 'MSF_y_10', 'MSF_y_inf', 'MSF_clk', 'M_tclk']
        self.msg_tmp_cols = ['height_1', "height_2", "height_3", "MSG_tclk"]
        self.info_cols = ['userId', 'postId',  "newsletter", "title", 'timestamp', "index", "read"]

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        X_train1 = X_train[self.msg_tmp_cols].values
        X_train2 = X_train[self.user_pattern_cols].values
        X_val1 = X_val[self.msg_tmp_cols].values
        X_val2 = X_val[self.user_pattern_cols].values
        y_train = y_train.values
        y_val = y_val.values

        user_pattern_input = Input(shape=(len(self.user_pattern_cols),))
        user_msg_input = Input(shape=(len(self.msg_tmp_cols),))

        x = Dense(16, activation="relu")(user_msg_input)
        x = Dense(8, activation="relu")(x)
        x = Model(inputs=user_msg_input, outputs=x)

        y = Dense(16, activation="relu")(user_pattern_input)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=user_pattern_input, outputs=y)
        combined = tf.keras.layers.Multiply()([x.output, y.output])
        z = Dense(1)(combined)
        model = Model(inputs=[x.input, y.input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=opt)
        print("[INFO] training model...")
        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32'), X_train2.astype('float32')], y=y_train,
            sample_weight=np.asarray([20.0 if y > 0 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32'), X_val2.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        model.save(self.save_path + self.name + ".h5")
        self.model = model

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict([df[self.msg_tmp_cols].values.astype('float32'),
                                    df[self.user_pattern_cols].values.astype('float32')])
        preds = np.exp(preds)
        X_test["pred"] = preds
        X_test["target"] = y_test
        pred_s = X_test.groupby(by=["userId", "postId", "timestamp"])["pred"].sum().reset_index()
        X_test = pd.merge(X_test, pred_s, on=["userId", "postId", "timestamp"])
        X_test["pred"] = X_test["pred_x"] / X_test["pred_y"]

        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class AlgoModel:
    def __init__(self, i):
        self.i = i

    def predict(self, X):
        return X["height_" + str(self.i)]


class Baseline:
    def __init__(self, i):
        self.name = "Baseline" + str(i)
        self.i = i

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = AlgoModel(self.i)

    def evaluate(self, X_test, y_test):
        df = X_test
        preds = self.model.predict(df)
        X_test["pred"] = preds
        X_test["target"] = y_test
        t_preds = X_test.groupby(by=["userId", "postId", "index"])[["pred", "target"]].sum()
        n_word = X_test.groupby(by=["userId", "postId", "index"])["n_word"].first()
        t_preds = t_preds.join(n_word)
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[(t_preds["target"] >= 10)].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()

        print(res, res2, c)
        return res, res2, c


class PatternSessionNNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = 'PatternSessionNNModel'
        self.dir_path = dir_path
        self.user_pattern_cols = ['MMF_y_2', 'MMF_y_5', 'MMF_y_10', 'MMF_y_inf', 'MMF_x_2', 'MMF_x_5', 'MMF_x_10',
                                  'MMF_x_inf',
                                  'MSF_y_2', 'MSF_y_5', 'MSF_y_10', 'MSF_y_inf', 'MSF_clk']
        self.user_session_cols = ['MMF_y_inf', 'MMF_x_inf', 'MSF_y_inf', 'MSF_clk']
        self.user_tmp_cols = ['isVisible', "mouseX", "mouseY",
                              'M_tclk', 'S_cy', 'S_h']
        self.msg_tmp_cols = ['MSG_y', "MSG_h", "MSG_tclk"]
        self.msg_session_cols = ['time_1', 'time_2', 'time_3', 'MSG_tmy', 'MSG_sh', 'MSG_sy', 'MSG_clk', 'MSG_svt']
        self.info_cols = ['userId', 'postId', "newsletter", "title", 'timestamp', "index", "n_word"]

    def load_data(self, filename):
        filename = self.dir_path + filename
        df = pd.read_csv(filename)
        return df

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test, filename="session_feature_labeled.csv"):
        df = self.load_data(filename)
        X = df[self.info_cols + self.user_session_cols + self.msg_session_cols]
        y_read_train = X_train.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        y_read_val = X_val.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        y_read_test = X_test.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        train_match = X_train.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        val_match = X_val.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        test_match = X_test.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        X_train = pd.merge(X, train_match, on=["userId", "postId", "index"], how='inner')
        X_val = pd.merge(X, val_match, on=["userId", "postId", "index"], how='inner')
        X_test = pd.merge(X, test_match, on=["userId", "postId", "index"], how='inner')
        X_train = X_train[[col for col in X_train.columns if col != 'read']]
        X_val = X_val[[col for col in X_train.columns if col != 'read']]
        X_test = X_test[[col for col in X_test.columns if col != 'read']]
        y_train = pd.merge(X_train, y_read_train, on=["userId", "postId", "index"], how='left')["read"]
        y_val = pd.merge(X_val, y_read_val, on=["userId", "postId", "index"], how='left')["read"]
        y_test = pd.merge(X_test, y_read_test, on=["userId", "postId", "index"], how='left')["read"]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
        X_train1 = X_train[self.user_session_cols].values
        X_train2 = X_train[self.msg_session_cols].values
        X_val1 = X_val[self.user_session_cols].values
        X_val2 = X_val[self.msg_session_cols].values
        y_train = y_train.values
        y_val = y_val.values
        user_pattern_input = Input(shape=(len(self.user_session_cols),))
        user_msg_input = Input(shape=(len(self.msg_session_cols),))
        x = Dense(16, activation="relu")(user_pattern_input)
        x = Dense(8, activation="relu")(x)
        x = Model(inputs=user_pattern_input, outputs=x)

        y = Dense(16, activation="relu")(user_msg_input)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=user_msg_input, outputs=y)
        combined = tf.keras.layers.Multiply()([x.output, y.output])
        z = Dense(1, activation="relu")(combined)
        model = Model(inputs=[x.input, y.input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss="mean_absolute_error", optimizer=opt)
        print("[INFO] training model...")

        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32'), X_train2.astype('float32')], y=y_train,
            sample_weight=np.asarray([5.0 if y >= 30 else 1.0 for y in y_train]),
            validation_data=([X_val1.astype('float32'), X_val2.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        self.model = model
        model.save(self.save_path + self.name + ".h5")

    def transform(self, y):
        x = [[0, 0, 0] for _ in range(len(y))]
        for i, t in enumerate(y):
            x[i][int(t)] = 1
        return np.asarray(x)

    def evaluate(self, X_test, y_test):
        df = self.X_test
        y_test = self.y_test
        df1 = df[self.user_session_cols].values
        df2 = df[self.msg_session_cols].values
        preds = self.model.predict([df1, df2])
        df["pred"] = np.round(preds)
        df["target"] = y_test

        t_preds = df[["pred", "target", "n_word"]]
        t_preds["read_level"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["target"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["target"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        t_preds["read_level_pred"] = ((t_preds["n_word"] > 20).astype(int)) * (
                    (t_preds["pred"] > (t_preds["n_word"] / 200 * 60)).astype(int) + (
                        t_preds["pred"] > (t_preds["n_word"] / 400 * 60)).astype(int))
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        res = ((np.round(t_preds["pred"]) - t_preds["target"]) / t_preds["target"]).abs()
        res.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = res[t_preds["target"] >= 10].mean()
        res2 = (np.round(t_preds["pred"]) - t_preds["target"]).abs()
        res2.replace([np.inf, -np.inf], np.nan, inplace=True)
        res2 = res2[t_preds["target"] < 10].mean()
        print(res, res2, c)
        return res, res2, c


class PatternSessionCatNNModel:
    def __init__(self, dir_path, save_path):
        self.save_path = save_path
        self.name = 'PatternSessionCatNNModel'
        self.dir_path = dir_path
        self.user_pattern_cols = ['MMF_y_2', 'MMF_y_5', 'MMF_y_10', 'MMF_y_inf', 'MMF_x_2', 'MMF_x_5', 'MMF_x_10',
                                  'MMF_x_inf',
                                  'MSF_y_2', 'MSF_y_5', 'MSF_y_10', 'MSF_y_inf', 'MSF_clk']
        self.user_session_cols = ['MMF_y_inf', 'MMF_x_inf', 'MSF_y_inf', 'MSF_clk']
        self.user_tmp_cols = ['isVisible', "mouseX", "mouseY",
                              'M_tclk', 'S_cy', 'S_h']
        self.msg_tmp_cols = ['MSG_y', "MSG_h", "MSG_tclk"]
        self.msg_session_cols = ['time_1', 'time_2', 'time_3', 'MSG_tmy', 'MSG_sh', 'MSG_sy', 'MSG_clk', 'MSG_svt']
        self.info_cols = ['userId', 'postId', "newsletter", "title", 'timestamp', "index", "read_level",
                          "n_word"]

    def load_data(self, filename):
        filename = self.dir_path + filename
        df = pd.read_csv(filename)
        return df

    def train(self, X_train, X_val, X_test, y_train, y_val, y_test, filename="session_feature_labeled.csv"):
        df = self.load_data(filename)
        X = df[self.info_cols + self.user_session_cols + self.msg_session_cols]
        y_read_train = X_train.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        y_read_val = X_val.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        y_read_test = X_test.groupby(by=["userId", "postId", "index"])["read"].sum().to_frame()
        train_match = X_train.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        val_match = X_val.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        test_match = X_test.groupby(["userId", "postId", "index"]).size().reset_index(name='Freq')[
            ["userId", "postId", "index"]]
        X_train = pd.merge(X, train_match, on=["userId", "postId", "index"], how='inner')
        X_val = pd.merge(X, val_match, on=["userId", "postId", "index"], how='inner')
        X_test = pd.merge(X, test_match, on=["userId", "postId", "index"], how='inner')

        X_train = X_train[[col for col in X_train.columns if col != 'read']]
        X_val = X_val[[col for col in X_train.columns if col != 'read']]
        X_test = X_test[[col for col in X_test.columns if col != 'read']]
        y_train = pd.merge(X_train, y_read_train, on=["userId", "postId", "index"], how='left')["read"]
        y_train = ((X_train["n_word"] > 20).astype(int)) * ((y_train > (X_train["n_word"] / 200 * 60)).astype(int) + (
                    y_train > (X_train["n_word"] / 400 * 60)).astype(int))
        y_val = pd.merge(X_val, y_read_val, on=["userId", "postId", "index"], how='left')["read"]
        y_val = ((X_val["n_word"] > 20).astype(int)) * (
                    (y_val > (X_val["n_word"] / 200 * 60)).astype(int) + (y_val > (X_val["n_word"] / 400 * 60)).astype(
                int))
        y_test = pd.merge(X_test, y_read_test, on=["userId", "postId", "index"], how='left')["read"]
        y_test = ((X_test["n_word"] > 20).astype(int)) * ((y_test > (X_test["n_word"] / 200 * 60)).astype(int) + (
                    y_test > (X_test["n_word"] / 400 * 60)).astype(int))
        y_train = self.transform(y_train)
        y_val = self.transform(y_val)
        y_test = self.transform(y_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
        X_train1 = X_train[self.user_session_cols].values
        X_train2 = X_train[self.msg_session_cols].values
        X_val1 = X_val[self.user_session_cols].values
        X_val2 = X_val[self.msg_session_cols].values

        user_pattern_input = Input(shape=(len(self.user_session_cols),))
        user_msg_input = Input(shape=(len(self.msg_session_cols),))
        x = Dense(16, activation="relu")(user_pattern_input)
        x = Dense(8, activation="relu")(x)
        x = Model(inputs=user_pattern_input, outputs=x)
        y = Dense(16, activation="relu")(user_msg_input)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=user_msg_input, outputs=y)
        combined = tf.keras.layers.Multiply()([x.output, y.output])
        z = Dense(3, activation="softmax")(combined)
        model = Model(inputs=[x.input, y.input], outputs=z)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits=False), optimizer=opt)
        print("[INFO] training model...")

        callbacks = [EarlyStopping(monitor='val_loss')]
        model.fit(
            x=[X_train1.astype('float32'), X_train2.astype('float32')], y=y_train,
            sample_weight=np.asarray([5 if y[0] == 0 else 1 for y in y_train]),
            validation_data=([X_val1.astype('float32'), X_val2.astype('float32')], y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print("[INFO] predicting read...")
        self.model = model
        model.save(self.save_path + self.name + ".h5")

    def transform(self, y):
        x = [[0, 0, 0] for _ in range(len(y))]
        for i, t in enumerate(y):
            x[i][int(t)] = 1
        return np.asarray(x)

    def transform_back(self, y_list):
        res = []
        for y in y_list:
            res.append(np.argmax(y))
        return res

    def evaluate(self, X_test, y_test):
        df = self.X_test
        y_test = self.y_test
        df1 = df[self.user_session_cols].values
        df2 = df[self.msg_session_cols].values
        preds = self.model.predict([df1, df2])
        df["read_level_pred"] = np.round(self.transform_back(preds))
        df["read_level"] = self.transform_back(y_test)
        t_preds = df[["read_level_pred", "read_level"]]
        c = confusion_matrix(t_preds["read_level"], t_preds["read_level_pred"])
        return 0, 0, c


class EyetrackingModel():
    def __init__(self, dir_path, save_path):
        self.dir_path = dir_path
        self.save_path = dir_path + save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        original_stdout = sys.stdout
        with open(self.save_path + 'param.txt', 'w') as convert_file:
            sys.stdout = convert_file
            print("epochs", epochs)
            print("batch_size", batch_size)
            print("lr", lr)
            print("seed", SEED)
            sys.stdout = original_stdout

    def load_data(self, filename):
        filename = self.dir_path + filename
        df = pd.read_csv(filename)
        print(filename, df.shape)
        return df

    def overall_train(self, filename="second_feature_labeled.csv", target="read",
                      filename2="session_feature_labeled.csv",
                      target2="read_level"):
        df = self.load_data(filename)

        df[target] = df[target].fillna(0)

        df2 = self.load_data(filename2)
        df2[target2] = df2[target2].fillna(0)
        X = df[[col for col in df.columns if col != 'target']]
        y = df[target]
        userIds = X.userId.unique()
        n = len(userIds)
        model_perf = {}
        model_time_perf = {}
        res = pd.DataFrame(
            columns=["round", "algo", "per_error", "abs_error", "accuracy", "skim_precision", "skim_recall",
                     "detail_precision", "detail_recall", "read_precision", "read_recall"])
        for i, userId in enumerate(userIds):
            print(i, n)

            for j in range(8):
                X_train = X.loc[X.userId != userId]
                X_test = X.loc[X.userId == userId]
                y_test = y.loc[X_test.index]

                train_userIds = X_train.userId.unique()
                X_vals = []
                for train_userId in train_userIds:
                    val_postIds = X_train.loc[X_train.userId == train_userId, "postId"].unique()
                    if j < len(val_postIds):
                        val_postId = val_postIds[j]
                    else:
                        val_postId = \
                        X_train.loc[X_train.userId == train_userId, "postId"].sample(n=1, random_state=SEED).iloc[0]
                    print(val_postId)
                    X_vals.append(X_train.loc[(X_train.userId == train_userId) & (X_train.postId == val_postId)])
                X_val = pd.concat(X_vals, axis=0)
                X_train = X_train.drop(X_val.index)
                y_train = y.loc[X_train.index]
                y_val = y.loc[X_val.index]

                print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

                dir_path = self.dir_path
                lg = LogisticModel(dir_path, self.save_path)
                st = PatternNNModel(dir_path, self.save_path)
                stnopat = NNModel(dir_path, self.save_path)
                at = PatternBaselineNNModel(dir_path, self.save_path)
                atnopat = BaselineNNModel(dir_path, self.save_path)
                a1 = Baseline(1)
                a2 = Baseline(2)
                a3 = Baseline(3)
                sst = PatternSessionNNModel(dir_path, self.save_path)
                sstc = PatternSessionCatNNModel(dir_path, self.save_path)
                model_dict = {
                    lg.name: lg,
                    stnopat.name: stnopat,
                    st.name: st,
                    atnopat.name: atnopat,
                    at.name: at,
                    a1.name: a1,
                    a2.name: a2,
                    a3.name: a3,
                    sst.name: sst,
                    sstc.name: sstc
                }
                for key in model_dict.keys():
                    print(key)
                    model_dict[key].train(X_train, X_val, X_test, y_train, y_val, y_test)
                    tmp = model_dict[key].evaluate(X_test, y_test)
                    res.loc[res.shape[0]] = [i * 8 + j, key, tmp[0], tmp[1], np.sum(np.trace(tmp[2])) / np.sum(tmp[2]),
                                             tmp[2][1][1] / np.sum(tmp[2][:, 1]), tmp[2][1][1] / np.sum(tmp[2][1, :]),
                                             tmp[2][2][2] / np.sum(tmp[2][:, 2]), tmp[2][2][2] / np.sum(tmp[2][2, :]),
                                             np.sum(tmp[2][1:, 1:]) / np.sum(tmp[2][:, 1:]),
                                             np.sum(tmp[2][1:, 1:]) / np.sum(tmp[2][1:, :])
                                             ]
                    if key not in model_time_perf:
                        model_time_perf[key] = [tmp[0] / len(userIds) / 8, tmp[1] / len(userIds) / 8, tmp[2]]
                    else:
                        model_time_perf[key][0] += tmp[0] / len(userIds) / 8
                        model_time_perf[key][1] += tmp[1] / len(userIds) / 8
                        model_time_perf[key][2] += tmp[2]
                    print(key, tmp)
                print(model_perf, model_time_perf)
        original_stdout = sys.stdout
        with open(self.save_path + 'nfold_result_summary.txt', 'w') as convert_file:
            sys.stdout = convert_file
            print(model_time_perf)
            sys.stdout = original_stdout
        res.to_csv(self.save_path + 'nfold_result.csv', index=False)
        tmp = res.groupby("algo").mean()
        print(tmp)
        tmp.to_csv(self.save_path + 'nfold_result_avg.csv')

    def output(self, filename="second_feature_labeled.csv", target="read", filename2="session_feature_labeled.csv",
               target2="read_level"):
        df = self.load_data(filename)
        df[target] = df[target].fillna(0)

        df2 = self.load_data(filename2)
        df2[target2] = df2[target2].fillna(0)
        df2 = df2[["userId", "postId", "index"]]
        df = pd.merge(df, df2, how="inner", on=["userId", "postId", "index"])
        X = df[[col for col in df.columns if col != 'target']]
        y = df[target]
        X_train = X
        X_test = X
        y_train = y
        y_test = y
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        dir_path = self.dir_path
        # st = SecondTrainer(dir_path, self.save_path)
        # stnopat = SecondTrainerNoPat(dir_path, self.save_path)
        at = PatternBaselineNNModel(dir_path, self.save_path)
        # atnopat = AlgoTrainerNoPat(dir_path, self.save_path)
        # a1 = Algo(1)
        # a2 = Algo(2)
        # a3 = Algo(3)
        # sst = SessionTrainer(dir_path, self.save_path)
        # sstc = SessionCatTrainer(dir_path, self.save_path)
        model_dict = {
            # st.name: st,
            # stnopat.name: stnopat,
            at.name: at,
            # atnopat.name: atnopat,
            # a1.name: a1,
            # a2.name: a2,
            # a3.name: a3,
            # sst.name: sst,
            # sstc.name: sstc
        }

        for key in model_dict.keys():
            print(key)
            model_dict[key].train(X_train, X_test, X_test, y_train, y_test, y_test)


if __name__ == "__main__":
    dir_path = "../data/"
    model = EyetrackingModel(dir_path, save_path="saved_result/")
    model.overall_train()
    # uncomment to train on all the data
    # model.output()
