# train_gesture_tflm.py
import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ===== Paths =====
CAPTURE_DIR = "captures"             # where your CSVs are saved
OUT_TFLITE  = "gesture_int8.tflite"  # quantized model
OUT_HEADER  = "gesture_model.h"      # C header for Arduino

# ===== Hyper/Shapes =====
NUM_SAMPLES = 100       # rows per window (matches your firmware)
FEATS       = ["ax","ay","az","gx","gy","gz"]
N_FEATS     = len(FEATS)
HIDDEN_UNITS = 32       # tiny dense layer

# ----- Load CSVs and build windows -----
def load_windows():
    X, y = [], []
    csvs = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {CAPTURE_DIR}/")

    for path in csvs:
        df = pd.read_csv(path)
        # Basic checks
        needed = set(["label","rep","idx"] + FEATS)
        if not needed.issubset(df.columns):
            print(f"Skipping {path}: missing columns")
            continue

        # One window per (label,rep)
        for (label, rep), grp in df.groupby(["label","rep"]):
            grp = grp.sort_values("idx")
            if len(grp) != NUM_SAMPLES:
                # Skip incomplete reps
                continue
            X.append(grp[FEATS].to_numpy(dtype=np.float32))  # shape (100, 6)
            y.append(int(label) - 1)  # to 0/1

    X = np.stack(X)                    # (N, 100, 6)
    y = np.array(y, dtype=np.int32)    # (N,)
    return X, y

print("Loading data...")
X, y = load_windows()
print("Dataset:", X.shape, y.shape, "classes:", np.unique(y))

# ----- Train/val split -----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Build tiny model: Flatten -> Dense(16, relu) -> Dense(2) -> Softmax -----
inp = tf.keras.Input(shape=(NUM_SAMPLES, N_FEATS), name="window")
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
logits = tf.keras.layers.Dense(2)(x)
out = tf.keras.layers.Softmax()(logits)
model = tf.keras.Model(inp, out)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
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

# ----- INT8 full-integer quantization -----
def rep_data():
    # Representative dataset for int8 calibration
    for i in range(min(200, len(X_train))):
        # yield a batch-sized sample with the same input shape
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

# ----- Make C header with the model bytes -----
def bytes_to_carray(b, varname):
    lines = []
    line = []
    for i, by in enumerate(b):
        line.append(str(by))
        if (i+1) % 12 == 0:
            lines.append(",".join(line))
            line = []
    if line:
        lines.append(",".join(line))
    body = ",\n  ".join(lines)
    return f"""#pragma once
#include <stdint.h>
extern const unsigned char {varname}[];
extern const unsigned int {varname}_len;
"""

def write_header(tflite_bytes, header_path, varname="g_gesture_model_data"):
    # Write header and a matching .cc file next to it
    base = os.path.splitext(header_path)[0]
    header_guard = os.path.basename(header_path).upper().replace(".", "_")
    # Header
    with open(header_path, "w") as hf:
        hf.write(f"#pragma once\n#include <stdint.h>\n")
        hf.write(f"extern const unsigned char {varname}[];\n")
        hf.write(f"extern const unsigned int {varname}_len;\n")
    # Source
    cc_path = base + ".cc"
    with open(cc_path, "w") as cf:
        cf.write("#include <cstddef>\n")
        cf.write(f'extern const unsigned char {varname}[] = {{\n  ')
        # Emit bytes
        for i, by in enumerate(tflite_bytes):
            cf.write(str(by))
            cf.write("," if i < len(tflite_bytes)-1 else "")
            if (i+1) % 12 == 0:
                cf.write("\n  ")
        cf.write("\n};\n")
        cf.write(f"extern const unsigned int {varname}_len = {len(tflite_bytes)};\n")
    print("Wrote", header_path, "and", cc_path)

write_header(tflite_model, OUT_HEADER)
print("Done.")
