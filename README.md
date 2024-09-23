# acquisition_software

acquisition tools for MCC DAQ and FT232H I2C data acquisition.

- http://mielke-bio.info/falk/posts/01.fpequipment
- http://mielke-bio.info/falk/posts/02.fpcalculations

# MCC DAQ drivers and library
- follow readme in https://github.com/mccdaq/uldaq
    - find latest release of mccdaq/uldaq (`v1.2.1` at time of testing)
    - download  `wget -N https://github.com/mccdaq/uldaq/releases/download/v1.2.1/libuldaq-1.2.1.tar.bz2`
    - extract   `tar -xvjf libuldaq-1.2.1.tar.bz2 && cd libuldaq-1.2.1`
    - build     `./configure && make -j2 && sudo make install -j2`
    - if "make" fails, you might need to do: ln -s /usr/bin/autom-1.16 /usr/bin/aclocal-1.14 && ln -s /usr/bin/automake-1.16 /usr/bin/automake-1.14
- install uldaq python bindings: `pip install uldaq`

