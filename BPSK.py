import numpy as np

class BPSK:
    def modulate(self, bits):
        bpsk_symbols = [1 if bit else -1 for bit in bits]
        if len(bits) % 48 != 0:
            bpsk_symbols += [0] * (48 - (len(bits) % 48))
        ofdm_symbols = []

        bpsk_symbols = np.reshape(bpsk_symbols, (-1, 48))
        self.data_indices = list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)) + \
                     list(range(38, 43)) + list(range(44, 57)) + list(range(58, 64))
        pilot_indices = [7, 21, 43, 57]

        ofdm_symbols = np.zeros((bpsk_symbols.shape[0], 64), dtype = complex)
        ofdm_symbols[:, self.data_indices] = bpsk_symbols
        ofdm_symbols[:, pilot_indices] = np.tile([1], (bpsk_symbols.shape[0], 4))

        return ofdm_symbols

    def demodulate(self, ofdm_symbols):
        bpsk_symbols = ofdm_symbols[:, self.data_indices].flatten()
        return [1 if np.real(s) >= 0 else 0 for s in bpsk_symbols]
 
