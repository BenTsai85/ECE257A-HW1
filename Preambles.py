import numpy as np
import math
import matplotlib.pyplot as plt

class Preamble:
    def stf_modulate(self, data):
        stf_f = np.zeros(64, dtype = complex)
        stf_f[-26:] = math.sqrt(13/6) * np.array([
            0, 0, 1+1j, 0, 0, 0, -1-1j, 0,
            0, 0, 1+1j, 0, 0, 0, -1-1j, 0,
            0, 0, -1-1j, 0, 0, 0, 1+1j, 0,
            0, 0
        ])
        stf_f[:27] = math.sqrt(13/6) * np.array([
            0, 0, 0, 0, -1-1j, 0,
            0, 0, -1-1j, 0, 0, 0, 1+1j, 0,
            0, 0, 1+1j, 0, 0, 0, 1+1j, 0,
            0, 0, 1+1j, 0, 0
        ])

        rshort = np.zeros(64, dtype = complex)
        for n in range(64):
            for k in range(-24, 25):
                rshort[n] += stf_f[k + 25] * np.exp(2j * math.pi * k / 64)


        stf_t = np.fft.ifft(rshort)
        self.stf_t = stf_t[:16]
        repeated_stf_t = np.tile(self.stf_t, 10)

        plt.figure()
        for i in range(0, data.shape[0], 80):
            plt.plot(list(range(-40, 40)), 10 * np.log10(np.square(np.abs(np.fft.fft(data[i : i + 80])))))
        plt.savefig("1-1.png")
        plt.figure()
        plt.plot(np.abs(repeated_stf_t))
        plt.savefig("1-2.png")

        return np.concatenate((repeated_stf_t, data))

    def ltf_modulate(self, data):
        self.ltf_f = np.zeros(64, dtype = complex)
        self.ltf_f[-26:] = np.array([
            1, 1, -1, -1, 1, 1, -1, 1,
            -1, 1, 1, 1, 1, 1, 1, -1,
            -1, 1, 1, -1, 1, -1, 1, 1,
            1, 1
        ])
        self.ltf_f[:27] = np.array([
            0, 1, -1, -1, 1, 1, -1, 1,
            -1, 1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, 1, -1, 1, -1, 1,
            1, 1, 1
        ])

        ltf_t = np.fft.ifft(self.ltf_f)
        return np.concatenate((ltf_t[-32:], ltf_t, ltf_t, data))

    def demodulate(self, data, stf_start):
        return data[stf_start + 320:]