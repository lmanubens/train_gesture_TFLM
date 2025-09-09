from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
import sys

# Map BuiltinOperator enum values to names (manually copied from TFLite schema)
OP_NAMES = {v: k for k, v in BuiltinOperator.__dict__.items() if isinstance(v, int)}

# Load TFLite model
with open("gesture_int8.tflite", "rb") as f:
    buf = f.read()

model = Model.GetRootAsModel(buf, 0)

print("Operators used in the model:")
found_shape = False
for i in range(model.OperatorCodesLength()):
    opcode = model.OperatorCodes(i)
    builtin_code = opcode.BuiltinCode()
    op_name = OP_NAMES.get(builtin_code, f"UNKNOWN({builtin_code})")
    print(f"{i+1}. {op_name}")
    if op_name == "SHAPE":
        found_shape = True

# Final verdict
if found_shape:
    print("\n❌ Model includes unsupported op: SHAPE")
else:
    print("\n✅ No SHAPE op found. Model is likely TFLM-compatible.")
