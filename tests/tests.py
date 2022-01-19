import numpy as np
import pytest
import matplotlib.pyplot as plt
from BlochHead import Pulse, Delay, Spin, ActualDelay, ActualPulse, BlochHead

T1, T2 = 100, 40  # us


def test_delay():

    D1 = ActualDelay(np.array([0, 1, 0]), 1e3, T1=T1, T2=T2, time_step=1)

    Mx = np.zeros(len(D1.time))
    My = np.exp(-D1.time / T2)
    Mz = 1 - (1 - 0)*np.exp(-D1.time / T1)

    np.testing.assert_almost_equal(D1.M[:, 0], Mx)
    np.testing.assert_almost_equal(D1.M[:, 1], My)
    np.testing.assert_almost_equal(D1.M[:, 2], Mz)


def test_delay_offset():
    D1 = ActualDelay(np.array([0, 1, 0]), 400, T1=T1, T2=T2, offsets=1, time_step=1)
    ExpMxy = np.linalg.norm(D1.M[:, :2], axis=1)
    Mxy = np.exp(-D1.time / T2)
    np.testing.assert_almost_equal(Mxy, ExpMxy, decimal=6)


def test_delay_offsets():
    D1 = ActualDelay(np.array([0, 1, 0]), 200, T1=T1, T2=T2, offsets=[-1, 0, 1], time_step=0.1)

    # Test  Mx +/- 1 offsets are  equal and opposite and that Mx 0 offset is 0
    np.testing.assert_almost_equal(D1.M[0, :, 0] + D1.M[2, :, 0], D1.M[1, :, 0])

    # Test  My +/- 1 offsets are  equal and not opposite and that My 0 offset is equal to the exponential decay of T2
    np.testing.assert_almost_equal(D1.M[0, :, 1], D1.M[2, :, 1])
    np.testing.assert_almost_equal(D1.M[1, :, 1], np.exp(-D1.time / T2))

    # Test that all Mz vectors are equal to the T1 decay
    Mz = 1 - (1 - 0) * np.exp(-D1.time / T1)
    for i in range(3):
        np.testing.assert_almost_equal(D1.M[i, :, 2], Mz)

delay_interface_args = [{},
                        {'M0': [0, 1, 0]},
                        {'M0': [0, 0, -1], 'time_step': 0.1},
                        {'M0': [0, 1, 0], 'T1': 100, 'T2': 10},
                        {'M0': [0, 1, 0], 'offsets': np.array([-1, 0, 1])}]

@pytest.mark.parametrize('args', delay_interface_args)
def test_delay_interface(args):
    D1 = Delay(delay_time=0.5, **args)
    if args == {}:
        with pytest.raises(TypeError):
            ActualD1 = ActualDelay(**D1.__dict__)
    else:
        ActualD1 = ActualDelay(**D1.__dict__)


def test_actual_sequence():

    P1 = ActualPulse(pulse_time=0.016, flip=np.pi/2, offsets=0)
    D1 = ActualDelay(M0=P1.M[-1], delay_time=0.1, T1=0.2, time_step=P1.time_step, offsets=0)
    P2 = ActualPulse(pulse_time=0.016, flip=np.pi, M0=D1.M[-1].copy(), offsets=0)
    D2 = ActualDelay(M0=P2.M[-1].copy(), delay_time=0.1, T1=0.2, time_step=P1.time_step, offsets=0)

    M, time = [], [[0]]
    for i, Event in enumerate([P1, D1, P2, D2]):
        idx = 0 if i == 0 else 1
        time.append(Event.time[idx:] + time[-1][-1])
        M.append(Event.M[..., idx:, :])

    time = np.concatenate(time[1:])
    M = np.concatenate(M)

    with np.load('test_data/actual_sequence.npz') as f:
        tans = f['time']
        Mans = f['M']

    np.testing.assert_allclose(tans, time)
    np.testing.assert_allclose(Mans, M)


def test_BlochBasic():

    spin = Spin(T1=0.2, offsets=0)
    events = [Pulse(pulse_time=0.016, flip=np.pi/2),
              Delay(delay_time=0.1),
              Pulse(pulse_time=0.016, flip=np.pi),
              Delay(delay_time=0.1)]

    block = BlochHead(spin, events, )

    with np.load('test_data/actual_sequence.npz') as f:
        tans = f['time']
        Mans = f['M']

    print(np.argmax(np.abs(Mans - block.M)))
    np.testing.assert_allclose(tans, block.time)
    np.testing.assert_allclose(Mans, block.M)


def test_SechTanh():

    spin = Spin(T1=0.5, offsets=np.linspace(-30, 30, 21), pulse_offsets=True)
    events = [Pulse(pulse_time=0.300,
                    time_step=1e-3,
                    flip=np.pi,
                    shape='sech/tanh',
                    freq=[-40, 40],
                    beta=10),
              Delay(delay_time=1.0)]

    block = BlochHead(spin, events)
    ans = np.load('test_data/SechTanh.npy')
    np.testing.assert_almost_equal(block.M, ans)



def test_deer1():

    offsets = np.linspace(-1, 1, 11)
    spin = Spin(offsets=offsets, time_step=0.001)

    events = [Delay(delay_time=0.05),
              Pulse(pulse_time=0.025, flip=np.pi/2),
              Delay(delay_time=0.4),
              Pulse(pulse_time=0.05, flip=np.pi),
              Delay(delay_time=0.4),
              Delay(delay_time=0.6, shift=-18.5),
              Pulse(pulse_time=0.05, flip=np.pi),
              Delay(delay_time=0.63)]

    block = BlochHead(spin, events)
    ans = np.load('test_data/deer1.npy')
    np.testing.assert_almost_equal(block.M, ans)

def test_deer2():

    offsets = np.linspace(-1, 1, 11)
    spin = Spin(M0=[0, 1, 0], offsets=offsets, time_step=0.001)

    events = [Delay(delay_time=0.2),
              Delay(delay_time=0.4, shift=-18.5),
              Pulse(pulse_time=0.05, flip=np.pi),
              Delay(delay_time=0.6)]

    block = BlochHead(spin, events)

    ans = np.load('test_data/deer2.npy')
    np.testing.assert_almost_equal(block.M, ans)


def test_deer3():

    offsets = np.linspace(-1, 1, 11)
    spin = Spin(M0=[0, 1, 0], offsets=offsets, time_step=0.001)

    events = [Delay(delay_time=0.37),
              Delay(delay_time=0.23, shift=-18.5),
              Pulse(pulse_time=0.05, flip=np.pi),
              Delay(delay_time=0.6)]

    block = BlochHead(spin, events)

    ans = np.load('test_data/deer3.npy')
    np.testing.assert_almost_equal(block.M, ans)

# TODO: test for passing kwargs to Pulse and Delay events