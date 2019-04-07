#!/usr/bin/env python
from nude import Nude
from glob import glob
import time
import sys


imgs = [Nude(x) for x in glob(sys.argv[1])]

start = time.time()

for _ in range(10):
    for img in imgs:
        img.result = None
        img.parse()

end = time.time()
print('%.6f seconds' % (end - start))
