
import IOToolbox as IOT


pins = [5]
with IOT.Trogger(pins) as trog:
    try:
        print ('logging...')
        trog.Log()
    except KeyboardInterrupt as ki:
        trog.Quit()