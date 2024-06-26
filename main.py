import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from Filter_LMS import FilterLMS

fs, y = wavfile.read('noise_ref.wav')
fs, target = wavfile.read('orig.wav')
x = target + 3 * y

ORDER = int(fs * 0.01)
lms = FilterLMS(num_taps=ORDER)

out = []
errs = []
yhats = []

for i in range(0, len(x)):
    if i % fs == 0:
        print(i)
    yhat, err = lms.filter(y[i], x[i])
    yhats.append(yhat)
    out.append(x[i] - yhat)
    errs.append(err)

out_array = np.array((out - np.min(out))/(np.max(out) - np.min(out)), dtype=np.float32)

wavfile.write('cleaned.wav', fs, out_array)

fig, ax = plt.subplots(2)
ax[0].plot(x, label="noisy")
ax[0].plot(target, label="target")
ax[0].plot(out, label="cleaned")

ax[1].plot((target - out_array) ** 2)

plt.show()
