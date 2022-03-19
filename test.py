import sys
import numpy as np

import finger
import gesture


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 test.py <filepath of data>")
        sys.exit(-1)

    data = SensorData(sys.argv[1])

    xs = data.raw
    y = np.arange(10)

    for x in xs:

