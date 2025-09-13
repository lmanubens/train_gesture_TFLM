# train_gesture_tflm.py
import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ===== Paths =====
CAPTURE_DIR = "captures"             # where your CSVs are saved
OUT_TFLITE  = "gesture_int8.tflite"  # quantized model
OUT_HEADER  = "gesture_model.h"      # C header for Arduino
N_CLASSES   = 2

# ===== Hyper/Shapes =====
NUM_SAMPLES  = 100       # rows per window
FEATS        = ["ax","ay","az","gx","gy","gz"]
N_FEATS      = len(FEATS)
HIDDEN_UNITS = 32

# ----- Load CSVs and build windows -----
def load_windows():
    X, y = [], []
    csvs = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {CAPTURE_DIR}/")

    for path in csvs:
        df = pd.read_csv(path)
        needed = set(["label","rep","idx"] + FEATS)
        if not needed.issubset(df.columns):
            print(f"Skipping {path}: missing columns")
            continue

        for (label, rep), grp in df.groupby(["label","rep"]):
            grp = grp.sort_values("idx")
            if len(grp) != NUM_SAMPLES:
                continue
            X.append(grp[FEATS].to_numpy(dtype=np.float32))  # shape (100, 6)
            y.append(int(label) - 1)  # labels 1/2 → 0/1

    X = np.stack(X)                    # (N, 100, 6)
    y = np.array(y, dtype=np.int32)    # (N,)
    return X, y

print("Loading data...")
X, y = load_windows()
print("Dataset:", X.shape, y.shape, "classes:", np.unique(y))

# ----- Normalization -----
mean = X.mean(axis=(0,1), keepdims=True)
std  = X.std(axis=(0,1), keepdims=True) + 1e-6

X_train = (X - mean) / std

# ----- Train/val split -----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Build tiny model -----
inp = tf.keras.Input(shape=(NUM_SAMPLES, N_FEATS), name="window")
# Keep original reshape flattening
x = tf.keras.layers.Lambda(
    lambda x: tf.reshape(x, [-1, NUM_SAMPLES * N_FEATS]),
    name="reshape"
)(inp)
x = tf.keras.layers.Dense(
    HIDDEN_UNITS,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)
out = tf.keras.layers.Dense(N_CLASSES)(x)
model = tf.keras.Model(inp, out)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=2
)

val_acc = history.history["val_accuracy"][-1]
print("Validation accuracy:", val_acc)

# ----- INT8 quantization -----
def rep_data():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

print("Converting to int8 TFLite...")
tflite_model = converter.convert()
with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)
print("Wrote", OUT_TFLITE, "size:", len(tflite_model), "bytes")

# ===== Get quantization params =====
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
scale, zero_point = input_details["quantization"]
print("Input quantization scale:", scale)
print("Input quantization zero_point:", zero_point)

# ----- Export as C array (.h + .cc) -----
def write_header(tflite_bytes, header_path, varname="g_gesture_model_data"):
    base = os.path.splitext(header_path)[0]
    # Header
    with open(header_path, "w") as hf:
        hf.write("#pragma once\n#include <stdint.h>\n\n")
        hf.write(f"extern const unsigned char {varname}[];\n")
        hf.write(f"extern const unsigned int {varname}_len;\n")
    # Source
    cc_path = base + ".cc"
    with open(cc_path, "w") as cf:
        cf.write("#include <cstddef>\n#include <cstdint>\n\n")
        cf.write(f"const unsigned char {varname}[] = {{\n  ")
        for i, b in enumerate(tflite_bytes):
            cf.write(str(b))
            if i < len(tflite_bytes)-1:
                cf.write(",")
            if (i+1) % 12 == 0:
                cf.write("\n  ")
        cf.write("\n};\n")
        cf.write(f"const unsigned int {varname}_len = {len(tflite_bytes)};\n")
    print("Wrote", header_path, "and", cc_path)

write_header(tflite_model, OUT_HEADER)

# ----- Save normalization params -----
np.savez("normalization_params.npz", mean=mean, std=std)
print("Normalization mean:", mean.flatten())
print("Normalization std :", std.flatten())
print("Done.")
