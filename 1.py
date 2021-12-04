import pylab
import numpy as np
import json
import pathlib
import math

def f(x):
    return  (np.sin(3 * math.pi * x) ** 3) + ((x - 1)**2) * (1 + (np.sin (math.pi * 3))**2)

x_min = -10
x_max = 10
dx = 0.01

x = np.arange(x_min, x_max, dx)
y = f(x)

res = {
"data": [
{"x": float(x1), "y": float(y1)} for x1, y1 in zip(x, y)
]
}

path = pathlib.Path("results")
path.mkdir(exist_ok=True)
file = path / "result_task1.json"
out = file.open("w")
json.dump(res, out, indent=4)
out.close()


pylab.plot(x, y)
pylab.grid()
pylab.savefig("results/task1.png")
