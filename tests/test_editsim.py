from unittest import TestCase
import numpy as np
import pandas as pd
from scipy import sparse
from nose.tools import ok_, eq_
from ..spykesim import editsim

def genpoisson_spiketrain(rate, dt, duration):
    offset = duration
    events = np.cumsum(np.random.exponential(scale = 1 / rate, size = int(duration*rate + offset)))
    return np.round(events[np.logical_and(0 < events, events < duration)], -int(np.log10(dt)))
def genpoisson_spiketrains(nneurons, rate, dt, duration):
    spike_timings = np.array([], dtype = np.float)
    spike_neurons = np.array([], dtype = np.int)
    for n in range(nneurons):
        spike_train = genpoisson_spiketrain(rate, dt, duration)
        spike_timings = np.r_[spike_timings, spike_train]
        spike_neurons = np.r_[spike_neurons, n * np.ones_like(spike_train, dtype = np.int)]
    return pd.DataFrame({
        "neuronid": spike_neurons,
        "spiketime": spike_timings
    })

def gen_sequence(nneurons = 10, seqlen = 0.1, dt = 0.001):
    return np.round(np.linspace(dt, seqlen-dt, nneurons), int(-np.log10(dt)))

def gen_sequences(neurons = np.arange(10), nsequences = 10, start = 0, end = 600, seqlen = 0.1, dt = 0.001):
    spike_timings = np.array([], dtype = np.float)
    spike_neurons = np.array([], dtype = np.int)
    nneurons = len(neurons)
    sequence_onsets = np.arange(start, end - seqlen, seqlen)
    for onset in sequence_onsets:
        spike_timings = np.r_[spike_timings, onset + gen_sequence(nneurons, seqlen, dt)]
        spike_neurons = np.r_[spike_neurons, neurons]
    return pd.DataFrame({
        "neuronid": spike_neurons,
        "spiketime": spike_timings
    })
def df2binarray_csc(df, duration_ms = None, binwidth = 1):
    neuronids = df.neuronid
    spikes_ms = df.spiketime * 1000
    nneurons = int(neuronids.max()+1)
    nrow = nneurons
    if duration_ms:
        ncol = int(max(spikes_ms)) + 1
        print(ncol)
    else:
        ncol = int(duration_ms) // binwidth + 1000
    binarray_lil = sparse.lil_matrix((nrow, ncol))
    for neuronid in range(nneurons):
        spike_train_of_a_neuron = spikes_ms[neuronids == neuronid]
        bins = np.arange(0, ncol * binwidth, binwidth)
        digitized_spike_train_of_a_neuron = np.digitize(spike_train_of_a_neuron, bins) - 1
        binned_spike_train_of_a_neuron = np.bincount(digitized_spike_train_of_a_neuron)
        binarray_lil[neuronid, digitized_spike_train_of_a_neuron] = binned_spike_train_of_a_neuron[digitized_spike_train_of_a_neuron]
    return binarray_lil.tocsc()

class EditsimTestCase(TestCase):
    def setUp(self):
        dt = 0.001
        nsequences = 5
        seqlen = 0.1
        self.seqlen = seqlen
        nneurons = 100
        nseqkinds = 3
        df = pd.DataFrame()
        df_seq = gen_sequences(
            neurons = np.arange(10),
            nsequences = nsequences,
            start = 0,
            end = 10,
            seqlen = seqlen,
            dt = dt)
        df_seq = pd.DataFrame({
            "neuronid": df_seq.neuronid,
            "spiketime": np.copy(df_seq.spiketime)
        })
        df = pd.concat([df, df_seq])
        df_seq = gen_sequences(
            neurons = np.arange(10, 20),
            nsequences = nsequences,
            start = 10.1,
            end = 20,
            seqlen = seqlen,
            dt = dt)
        df_seq = pd.DataFrame({
            "neuronid": df_seq.neuronid,
            "spiketime": np.copy(df_seq.spiketime)
        })
        df = pd.concat([df, df_seq])
        df_seq = gen_sequences(
            neurons = np.arange(20, 30),
            nsequences = nsequences,
            start = 20.1,
            end = 30,
            seqlen = seqlen,
            dt = dt)
        df_seq = pd.DataFrame({
            "neuronid": df_seq.neuronid,
            "spiketime": np.copy(df_seq.spiketime)
        })
        df = pd.concat([df, df_seq])
        self.binmat = df2binarray_csc(df, 30)

    def test_simmat(self):
        window = int(self.seqlen*1000)
        a = 0.05
        es = editsim.FromBinMat(alpha=a)
        simmat_lsh = es.gensimmat(self.binmat, window, window, numband=1, bandwidth=50, minhash=True)

    def test_genidvec(self):
        eq_(1, 1)
