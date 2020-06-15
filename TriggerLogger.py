
import IOToolbox as IOT


pins = [3, 5]
with IOT.Trogger(pins) as trog:
    try:
        print ('logging...')
        trog.Log()
    except KeyboardInterrupt as ki:
        trog.Quit()