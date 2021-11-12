from numbers import Number
import numpy as np
from scipy.integrate import odeint
import PulseShape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

gamma_e = 1.760859644e2  # Electron Gyromagnetic Ratio in kRad/(us*T)
QbandFreq = 35e3         # MHz
QbandField = 1.2489e3    # Tesla


class Spin:

    def __init__(self, M0=np.array([0, 0, 1]), T1=np.inf, T2=np.inf, offsets=0, Meq=np.array([0, 0, 1]),
                 gyro=gamma_e, B0=QbandField, frame='rotating', **kwargs):

        """
        Spin system being used in the experiment
        :param M0:
        :param T1:
        :param T2:
        :param offsets:
        :param Meq:
        :param gyro:
        :param B0:
        :param frame:
        :param kwargs:
        """

        if T1 < T2:
            T2 = T1
        self.M0 = M0
        self.T1 = T1
        self.T2 = T2
        self.offsets = offsets
        self.time_step = kwargs.get('time_step', None)


        # Future stuff
        self.Meq = Meq
        self.gyro = gyro
        self.B0 = np.array([0, 0, B0]) if isinstance(B0, Number) else B0
        self.frame = frame


class Delay:

    def __init__(self, delay_time, **kwargs):
        """
        A Delay event object. Used to setup delays for a sequence in a pulsed experiment.Sequences of pulses and delays
        are used to create a BlochHead object which will simulate the experiment.

        :param delay_time: float
            The amount of time for the delay

        :param kwargs:
            Additional arguments that are passed to ActualDelay objects

        """
        self.delay_time = delay_time
        self.__dict__.update(kwargs)

class Pulse:

    def __init__(self, pulse_time, flip, **kwargs):
        """
        A Pulse event object. Used to setup pulses for a sequence in a pulsed experiment.Sequences of pulses and delays
        are used to create a BlochHead object which will simulate the experiment.

        :param pulse_time: float
            The amount of time over which the pulse is applied.

        :param flip: float
            The angle (in radians) the magnetization vector is flipped

        :param kwargs:
            Additional arguments that are  passed to ActualPulse objects
        """
        self.pulse_time = pulse_time
        self.flip = flip
        self.__dict__.update(kwargs)


class ActualDelay:
    def __init__(self, M0, delay_time, time_step=None, T1=np.inf, T2=np.inf, offsets=0):
        """
        ActualDelay is more of a backend object that is used by BlochHead to perform a Bloch sphere simulation. Creation
        of a ActualDelay object requires an M0 initial magnetization vector in addition to a delay_time. While
        ActualDelay objects are lower level objects they can still be used to create simulations without the BlochHead
        object and offer a little more customization.

        :param M0: numpy.ndarray
            D3 Vector or list/array of 3D vectors corresponding to the initial values of the magnetization vector or
            offsets at the beginning of the delay.

        :param delay_time: float  
        :param time_step:
        :param T1:
        :param T2:
        :param offsets:
        """
        if T1 < T2:
            T2 = T1
        offsets = np.atleast_1d(offsets)
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


class ActualPulse(PulseShape.Pulse):

    def __init__(self, pulse_time, time_step=None, flip=np.pi, mwFreq=33.80,
                 amp=None, Qcrit=None, freq=0, phase=0, type='rectangular',  **kwargs):

        super().__init__(pulse_time, time_step, flip, mwFreq, amp,
                         Qcrit, freq, phase, type, trajectory=True, **kwargs)

class BlochHead:

    def __init__(self, spin, sequence):
        self.M0 = spin.M0
        self.offsets = np.atleast_1d(spin.offsets)
        self.time = [[0]]
        self.M = [np.array([self.M0.copy() for i in range(len(self.offsets))])]
        if len(self.offsets) > 1:
            self.M[0] = self.M[0][:, None, :]

        self.time_step = spin.time_step

        for event in sequence:
            if self.time_step is not None:
                event.time_step = self.time_step

            if isinstance(event, Pulse):
                Event = ActualPulse(M0=self.M[-1][..., -1, :].copy(), offsets=self.offsets, **event.__dict__)
            elif isinstance(event, Delay):
                Event = ActualDelay(M0=self.M[-1][..., -1, :].copy(), offsets=self.offsets, **event.__dict__)
            else:
                raise ValueError('`sequence` must be an ordered container filled with Pulse and Delay objects')

            self.time.append(Event.time + self.time[-1][-1])
            self.M.append(Event.M)

            if self.time_step is None:
                self.time_step = Event.time_step

        self.time = np.concatenate(self.time[1:])

        self.M = np.concatenate(self.M, axis=-2)

    def save(self, filename='animation.mp4'):

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, cmap=plt.cm.viridis, alpha=0.2)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)

        if len(self.M.shape) > 2:
            quivers = [ax.quiver(0, 0, 0, *self.M[i, 0]) for i in range(len(self.offsets))]
        else:
            quivers = [ax.quiver(0, 0, 0, *self.M[0])]

        def update(t):
            nonlocal quivers
            for i, quiver in enumerate(quivers):
                quiver.remove()
                if len(self.M.shape) > 2:
                    quivers[i] = ax.quiver(0, 0, 0, *self.M[i, t])
                else:
                    quivers[i] = ax.quiver(0, 0, 0, *self.M[t])

        print('we made it this far and we have {len(self.time)} frames!')
        ani = FuncAnimation(fig, update, frames=np.arange(len(self.time)), interval=10)

        ani.save('animation.mp4')