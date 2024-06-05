import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from Filter_LMS import FilterLMS

fs, y = wavfile.read('noise_ref.wav')
fs, x = wavfile.read('noisy.wav')

ORDER = int(fs * 0.1)
lms = FilterLMS(num_taps=ORDER)

out = []
errs = []

for i in range(0, len(x)):
    if (i % fs == 0):
        print(i)
    yhat, err = lms.filter(x[i], y[i])
    out.append(x[i] - yhat)
    errs.append(err)

out_array = np.array((out - np.min(out))/(np.max(out) - np.min(out)), dtype=np.float32)

wavfile.write('cleaned.wav', fs, out_array)

fig, ax = plt.subplots(2)
ax[0].plot(x)
ax[0].plot(out)

ax[1].plot(errs)

plt.show()
