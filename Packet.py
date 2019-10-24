import numpy as np
import math
import matplotlib.pyplot as plt

from BPSK import BPSK
from OFDM import OFDM
from Preambles import Preamble
from Distortion import Distortion
        
class Packet:
    def __init__(self):
        self.length = 4160
        self.data = np.random.randint(2, size = self.length).tolist()
        self.bpsk = BPSK()
        self.ofdm = OFDM()
        self.preamble = Preamble()
        self.distortion = Distortion()
    def construction(self):
        self.bpsk_modulation()
        self.ofdm_modulation()
        self.add_preambles()
    def transmission(self):
        self.data = self.distortion.apply(self.data)
        self.plot_stf_samples()
        self.add_idle()
    def detection(self):
        self.r = np.zeros(self.data.shape[0])
        self.e = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0] - 31):
            self.r[i] = np.abs(np.dot(self.data[i : i + 16], np.conj(self.data[i + 16 : i + 32])))
            self.e[i] = np.sum(np.square(np.abs(self.data[i : i + 16])))
        self.plot_detection()
    def synchronization(self):
        cross_correlation = np.correlate(self.data, self.preamble.stf_t, mode = "same")[8:]
        self.plot_cross_correlation(np.abs(cross_correlation))
        stf_start = np.argwhere(np.abs(cross_correlation) > 0.9 * np.max(np.abs(cross_correlation)))
        print("Indices of STF starting time:\n{}".format(stf_start.flatten().tolist()))
        print("---------------------------------------------------------")
        self.stf_start = stf_start[0][0]
    def decoding(self):
        ltf_start = self.stf_start + 160
        # Calculate Frequency Offset
        self.data = self.distortion.recover_frequency_offset(self.data, ltf_start)
        # Calculate Channel Distortion H
        H = self.distortion.get_channel_distortion(self.data, self.preamble.ltf_f, ltf_start)
        print("Channel Distortion:\n{}".format(H.tolist()))
        print("---------------------------------------------------------")
        
        self.data = self.preamble.demodulate(self.data, self.stf_start)
        self.data = self.ofdm.demodulate(self.data)
        H = [1/i if i else 0 for i in H]
        self.data = np.multiply(self.data, H)
        self.data = self.bpsk.demodulate(self.data)
        # Remove padding
        self.data = self.data[:4160]

    def bpsk_modulation(self):
        self.data = self.bpsk.modulate(self.data)
    def ofdm_modulation(self):
        self.data = self.ofdm.modulate(self.data)
    def add_preambles(self):
        self.data = self.preamble.ltf_modulate(self.data)
        self.data = self.preamble.stf_modulate(self.data)
    def add_idle(self):
        self.data = np.concatenate((np.zeros(100), self.data))
    
    
    def plot_stf_samples(self):
        plt.figure()
        plt.plot(np.abs(self.data[:160]))
        plt.savefig("2.png")

    def plot_detection(self):
        plt.figure()
        plt.plot(self.r, label = "R")
        plt.plot(self.e, label = "E")
        plt.legend()
        plt.savefig("3.png")

    def plot_cross_correlation(self, cross_correlation):
        plt.figure()
        plt.plot(cross_correlation)
        plt.savefig("4.png")
