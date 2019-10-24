from Packet import Packet
import numpy as np

def bits2string(bits):
    bits = [''.join(map(str, bits[i : i + 8])) for i in range(0, len(bits), 8)]
    return ''.join([chr(int(x, 2)) for x in bits])

packet = Packet()
transmitted = np.array(packet.data)
print("Transmitted Strings:\n{}".format(bits2string(transmitted)))
print("---------------------------------------------------------")

packet.construction()
packet.transmission()
packet.detection()
packet.synchronization()
packet.decoding()

received = np.array(packet.data)
print("Received Strings:\n{}".format(bits2string(received)))
print("---------------------------------------------------------")
print("Error Rate: {}".format(np.sum(np.where(transmitted != received)) / transmitted.shape[0]))