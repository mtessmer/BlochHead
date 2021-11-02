from numbers import Number
import numpy as np
from scipy.integrate import odeint
import PulseShape

gamma_e = 1.760859644e2  # Electron Gyromagnetic Ratio in kRad/(us*T)
QbandFreq = 35e3         # MHz
QbandField = 1.2489e3    # Tesla


class SpinSystem:

    def __init__(self, M0, Meq=np.array([0, 0, 1]), gyro=gamma_e, B0=QbandField,
                 T1=np.inf, T2=np.inf, frame='rotating', offset=0):
        if T1 < T2:
            T2 = T1

        self.M0 = M0
        self.Meq = Meq
        self.gyro = gyro
        self.B0 = np.array([0, 0, B0]) if isinstance(B0, Number) else B0
        self.T1 = T1
        self.T2 = T2
        self.frame = frame
        self.offset = offset


class Delay:

    def __init__(self, M0, delay_time, time_step=None, T1=np.inf, T2=np.inf, offsets=0):
        """

        :param deay_time:
        :param M0:
        """
        if T1 < T2:
            T2 = T1

        self.T1 = T1
        self.T2 = T2

        M0 = np.asarray(M0, dtype=float)
        if len(M0.shape) == 1:
            M0 = np.tile(M0, (len(offsets), 1))

        self.time = np.arange(0, delay_time + np.finfo(float).eps, time_step)
        self.M = []
        offsets = np.atleast_1d(offsets)
        for i, offset in enumerate(offsets):
            self.dM = np.array([[  -1/T2, -offset,     0],
                                [offset,  -1/T2,     0],
                                [      0,      0, -1/T1]])

            self.M.append(odeint(self.dMdt, M0[i], self.time))

        self.M = np.squeeze(self.M)


    def dMdt(self, M, t):
        dM = self.dM @ M + np.array([0, 0, 1 / self.T1])
        return dM


class Pulse(PulseShape.Pulse):

    def __init__(self, pulse_time, time_step=None, flip=np.pi, mwFreq=33.80,
                 amp=None, Qcrit=None, freq=0, phase=0, type='rectangular',  **kwargs):

        super().__init__(pulse_time, time_step, flip, mwFreq, amp,
                         Qcrit, freq, phase, type, trajectory=True, **kwargs)

# class BlochHead:
#
#     def __init__(self, sequence, M0=np.arary([0, 0, 1]), offsets=None):
#
#         Traj = []
#         for event in sequence:
#             itraj = event.M
