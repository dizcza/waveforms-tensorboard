import io
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import torch
from neo.io import BlackrockIO
from neo.io.proxyobjects import SpikeTrainProxy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_reach_to_grasp_spiketrains():
    """
    Loads Rech-to-Grasp spiketrains.
    """
    datasets_dir = Path("datasets")
    assert datasets_dir.exists(), \
        "The 'datasets' folder with Reach-to-Grasp data " \
        "https://gin.g-node.org/INT/multielectrode_grasp must be located in " \
        "the same folder as this script."
    session_name = str(datasets_dir / "i140703-001")
    nev_override = str(datasets_dir / "i140703-001-03")
    session = BlackrockIO(filename=session_name, nsx_to_load=[],
                          nev_override=nev_override,
                          verbose=False)
    segment = session.read_segment(lazy=True)
    return segment.spiketrains


def extract_waveforms(spiketrains, per_unit=500, channels=[1], units='uV'):
    """
    Loads spike waveforms from a session of Reach-to-Grasp data.

    Parameters
    ----------
    spiketrains : list
        A list of `SpikeTrain` or `SpikeTrainProxy` (lazy mode) to
        extract the waveforms from.
    per_unit : int
        Num. of waveform samples to load per unit (neuron) for each channel.
        The samples are chosen randomly.
    channels : list
        A list of channel IDs to extract the waveforms from. All IDs must not
        succeed 96 - the total num. of channels in a Utah array.
    units : {'uV', 'mV'}
        The physical units to rescale the waveform amplitudes to.

    Returns
    -------
    waveforms : (N, 38) torch.Tensor
        2d array of `N` waveforms, each consisting of 38 time points.
    channel_units : (N,) list
        A list of (channel, unit) labels of size `N`.
    """
    waveforms = []
    channel_units = []
    for spiketrain in spiketrains:
        channel = int(spiketrain.annotations['channel_id'])
        unit_id = int(spiketrain.annotations['unit_id'])
        if channel not in channels:
            continue
        if isinstance(spiketrain, SpikeTrainProxy):
            # lazy mode
            spiketrain = spiketrain.load(time_slice=(10 * pq.s, 300 * pq.s),
                                         load_waveforms=True)
        waveform = spiketrain.waveforms
        wf_size = min(len(waveform), per_unit)
        select = np.random.choice(len(waveform), size=wf_size, replace=False)
        waveform = waveform[select, 0, :].rescale(units).magnitude
        waveform = torch.from_numpy(waveform)
        waveforms.append(waveform)
        if len(channels) == 1:
            label = unit_id
        else:
            label = (channel, unit_id)
        channel_units.extend([label] * wf_size)

    waveforms = torch.cat(waveforms, dim=0)
    return waveforms, channel_units


def _plot_single(waveform, ax):
    ax.clear()
    ax.tick_params(axis='y', direction='in', pad=-15)
    ax.plot(waveform)
    ax.set_xticks([])
    plt.tight_layout()
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = mpimg.imread(buf, format='png')[:, :, :3]  # skip alpha
    image = torch.from_numpy(image)
    return image


def create_figures(waveforms: torch.Tensor):
    """
    Creates images with matplotlib.

    Parameters
    ----------
    waveforms : (N, 38) torch.Tensor
        2d array of `N` waveforms, each consisting of 38 time points.

    Returns
    -------
    label_img : (N, 3, H, W) torch.Tensor
        An array of RGB figures for each input waveform.
    """
    waveforms = waveforms.numpy()

    plt.rcParams['figure.figsize'] = 0.8, 0.6
    plt.rcParams['font.size'] = 5
    ax = plt.axes(facecolor='#E6E6E6')
    label_images = [_plot_single(waveform, ax) for waveform in
                    tqdm(waveforms, desc="Creating figures")]
    label_images = torch.stack(label_images, dim=0)
    label_images = label_images.permute(dims=(0, 3, 1, 2))
    return label_images


def plot_embeddings():
    """
    The main function to visualize waveforms in TensorBoard.
    """
    spiketrains = load_reach_to_grasp_spiketrains()
    waveforms, channel_units = extract_waveforms(spiketrains, channels=[79])
    label_img = create_figures(waveforms)
    print(label_img.shape)
    if isinstance(channel_units[0], int):
        # only one channel
        label_names = None
    else:
        # multiple channels
        label_names = ['channel', 'unit']
    shutil.rmtree('runs', ignore_errors=True)
    writer = SummaryWriter('runs/R2G-waveforms')
    writer.add_embedding(waveforms, metadata=channel_units,
                         label_img=label_img,
                         metadata_header=label_names)
    writer.close()
    print("Done. Now open a terminal and run 'tensorboard --logdir=runs'")


if __name__ == '__main__':
    np.random.seed(9)
    plot_embeddings()
