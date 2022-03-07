import os
import sys

result_path = sys.argv[1]
for f in os.listdir(result_path):
    lines = []
    with open(os.path.join(result_path, f), "r") as res_f:
        lines += [line.split(",")[:-1] for line in res_f]
    os.remove(os.path.join(result_path, f))
    with open(os.path.join(result_path, f), "w") as res_f:
        res_f.write("\r\n".join([",".join(line) for line in lines]))
