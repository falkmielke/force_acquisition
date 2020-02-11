import sys as SYS
import IOToolbox as IOT


args = SYS.argv
with IOT.Oscilloscope( \
                  sampling_rate = 1e3 \
                , device_nr = 0 if (len(args) <= 1) else int(args[1]) \
                , channel_labels = IOT.forceplate_settings['dualkistler']['channel_order'] \
                , scan_frq = 1e6 \
                ) \
        as osci:

        osci.Show(window = 6)