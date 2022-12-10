#!/usr/bin/env python3
import sys
sys.path.insert(0,'/media/abanobsoliman/DATA/Abanob_PhD/Algorithms_Dev/Python_Alg_RGB-D-T/HCALIB/src/DSEC/scripts')
import argparse
from tqdm import tqdm
from pathlib import Path
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
from visualization.eventreader import EventReader
from utils.eventslicer import EventSlicer

def main():
    """Example reading of h5 events.
    EventReader and EventSlicer are python classes from the DSEC code base (https://github.com/uzh-rpg/DSEC).
    Publication of the DSEC datset can be found under (https://dsec.ifi.uzh.ch/):
        Mathias Gehrig, Willem Aarents, Daniel Gehrig and Davide Scaramuzza (2021).
        DSEC: A Stereo Event Camera Dataset for Driving Scenarios. IEEE Robotics and Automation Letters.
    """

    parser = argparse.ArgumentParser(description="Reading a h5 event file")
    parser.add_argument("--event_file", help="Events in 'events' are of type (x, y, t, p)"
                        "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                        default="//media/abanobsoliman/DATA/Abanob_PhD/Algorithms_Dev/Python_Alg_RGB-D-T/HCALIB/data/3Ms/mocap-6dof/mocap-6dof-events_left.h5")

    args = parser.parse_args()
    rawfile = Path(args.event_file)
    h5file = h5py.File(rawfile)
    print(h5file.keys())
    events = h5file['events']
    print("Contains %d events" % (events['t'].shape[0]))
    print("Event duration is %.2f seconds" % ((events['t'][-1] - events['t'][0])*1e-6))

    # Option1: Add your custom code here
    """
        Custom Code
    """

    # Option2: Alternatively, you can use the code from DSEC´s event data tool,
    # which for example allows to use an EventReader object for reading chunk-by-chunk
    dt_ms = 100
    for evs in tqdm(EventReader(rawfile, dt_ms)):
        print("keys: ", evs.keys())
        print("first timestamp (us) in dt_ms-batch: ", evs['t'][0])
        print("last timestamp (us) in dt_ms-batch: ", evs['t'][-1])
        break

    # DSEC´s tools also allow to use an EventSlicer object for reading in specific time interval
    slicer = EventSlicer(h5file)
    # get exemplary event batch between (15ms, 20ms)
    evs = slicer.get_events(20e3, 21e3)

    # Visualize the event batch as accumulated event image
    plt.figure()
    ev_arr = np.stack([evs['x'], evs['y'], evs['t'], evs['p']], axis=1)
    pos = ev_arr[np.where(ev_arr[:, 3] == 0)]
    neg = ev_arr[np.where(ev_arr[:, 3] == 1)]
    plt.scatter(pos[:, 0], pos[:, 1], color="blue", s=0.7)
    plt.scatter(neg[:, 0], neg[:, 1], color="red", s=0.7)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()

