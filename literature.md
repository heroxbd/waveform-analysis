1. Daya Bay
    1. Zeyuan Yu’s PhD thesis
        1. NULL
    2. PMT waveform modeling at the Daya Bay experiment
    3. The Flash ADC system and PMT waveform reconstruction for the Daya Bay experiment
        1. The first one was speed. The fitting of one waveform required about 0.5 s, which was a huge workload during the data reconstruction. The second one was fitting quality. The fitting had a increasing failure rate with increasing number of hits, introducing a residual nonlinearity which was difficult to calibrate. The waveform fitting method could be used in the crosscheck analysis of small event samples, for example, Inverse Beta Decays, etc., in which special care could be taken to examine the fitting quality.
        2. Besides the charge measurement, the algorithms had different timing separation abilities for pile-up hits. The waveform fitting and deconvolution could discriminate hits separated larger than 10 ns, while for the simple integral method it was 20 ns and for the Daya Bay CR-(RC)4 shaping 40 ns.
2. JUNO
    1. Particle Identification at MeV Energies in JUNO
        1. NULL
    2. Atmospheric neutrino spectrum reconstruction with JUNO
        1. NULL
    3. Comparison on PMT Waveform Reconstructions with JUNO Prototype
        1. Following the results for 1% overshoot ratio waveforms, we suggest that the simplest waveform integration can be considered for future fast preliminary reconstruction because the three algorithms give the same level results and considering that the setting of parameters, running and failure rate of waveform fitting is a big challenge, and it needs more work to build SPE model for huge number PMTs for deconvolution algorithm.
        2. pointed to: PMT waveform modeling at the Daya Bay experiment
    4. Capability of detecting low energy events in JUNO Central Detector
        1. NULL
    5. On the way to the determination of the Neutrino Mass Hierarchy with JUNO
        1. NULL
3. Borexino
    1. Neutrinos from the primary proton–proton fusion process in the Sun
        1. NULL
        2. pointed to: Final results of Borexino Phase-I on low-energy solar-neutrino spectroscopy
    2. Final results of Borexino Phase-I on low-energy solar-neutrino spectroscopy
        1. VI. ELECTRONICS AND TRIGGERS
        2. pointed to: The Borexino read out electronics and trigger system
        3. pointed to: A gateless charge integrator for Borexino energy measurement
    3. A gateless charge integrator for Borexino energy measurement
        1. The time interval needed to close the switches, perform the voltage reading and wait until the capacitor C*/5 is discharged is called here the dead time and it represents the time interval during which the system is not able to accept a new gate.
        2. The presence of this network spoils slightly the precision of the charge measurement of pulse sequencies separated by a time interval of the order of tint. 
    4. First Direct Experimental Evidence of CNO neutrinos
        1. The total number of detected photoelectrons and their arrival times are used to reconstruct the neutrino energy and its interaction point in the detector, respectively. 
4. LENA
    1. Topological track reconstruction in unsegmented, large-volume liquid scintillator detectors
        1. For experiments in which only the first photon hit time at each PMT is available, this method can, after some adaptations, still be useful to provide pure topological information. However, the information on the energy deposition in each point is largely lost in this case.
        2. Besides the timing, their main advantage is the small pixel size, which guarantees clearly defined single photon hits in contrast to a continuous waveform that is susceptible to complex pile-up effects from fast consecutive photon hits at large PMTs.
5. KamLAND
    1. Development of new trigger system for KamLAND2-Zen
        1. NULL
    2. Search for electron antineutrinos associated with gravitational wave events GW150914 and GW151226 using KamLAND
        1. NULL
        2. pointed to: Production of Radioactive Isotopes through Cosmic Muon Spallation in KamLAND
    3. Production of Radioactive Isotopes through Cosmic Muon Spallation in KamLAND
        1. However, the offline analysis takes full advantage of the information stored in the digitized PMT signals by identifying individual PMT pulses in the waveform information that is read out. The time and integrated area (called “charge”) are computed from the individual pulses. For each PMT, the average charge corresponding to a single p.e. is determined from single-pulse waveforms observed in low-occupancy events. 
        2. The ID muon track is reconstructed from arrival times of the first-arriving Cherenkov or scintillation photons at the PMTs. Since for relativistic muons the wavefront of the scintillation light proceeds at the Cherenkov angle, and since muons generate enough light to generate photoelectrons in every PMT, by restricting the fit to the first-arriving photons both Cherenkov and scintillation photons can be treated identically
    4. The KamLAND Full-Volume Calibration System
6. SNO
    1. Constraints on Neutrino Lifetime from the Sudbury Neutrino Observatory
        1. NULL
    2. Measurement of neutron production in atmospheric neutrino interactions at the Sudbury Neutrino Observatory
        1. NULL
        2. pointed to: Neutron Multiplicity in Atmospheric Neutrino Events at the Sudbury Neutrino Observatory
    3. Event Classification in Liquid Scintillator Using PMT Hit Patterns
        1. At the same time, the input signal is passed into two of the integration channels, one high gain and one low gain, which integrate over short and long time intervals (60 and 390 ns, respectively). If a global trigger arrives, the TAC stops and, after the integration time is complete, the charge and time (relative to trigger) is saved to analogue memory cells in the CMOS chip and a ‘CMOS data available’ flag is set.
        2. For example, events at the detector centre produce photons that all arrive at approximately the same time, but events near the AV create photons with a range of arrival times: the PMTs closest to the event are hit first and those on the far side of the detector hit last.
    4. Neutron Multiplicity in Atmospheric Neutrino Events at the Sudbury Neutrino Observatory
        1. The DB contained three kinds of custom application specific integrated circuits (ASICs) that applied a threshold to the PMT signal, integrated the charge of the pulse, and measured the relative time of the hits. 
7. EXO
    1. Event Reconstruction in a Liquid Xenon Time Projection Chamber with an Optically-Open Field Cage
        1. NULL no PMT
    2. Measurement of the scintillation and ionization response of liquid xenon at MeV energies in the EXO-200 experiment
        1. pointed to: Search for Neutrinoless Double-Beta Decay with the Upgraded EXO-200 Detector
    3. Search for Neutrinoless Double-Beta Decay with the Upgraded EXO-200 Detector
        1. NULL no PMT
8. XENON1T
    1. The XENON1T Data Acquisition System
        1. Because XENON1T is a rare event search, the full digitized waveform of every event must be recorded for later scrutiny, necessitating a high throughput capability in calibration modes. 
        2. pointed to: XENON1T Dark Matter Data Analysis: Signal Reconstruction, Calibration and Event Selection
    2. XENON1T Dark Matter Data Analysis: Signal Reconstruction, Calibration and Event Selection
        1. PMT signals exceeding a channel-specific threshold above the baseline, accepting on average 93% of SPE signals, are digitized at a rate of 108 samples/second by the data acquisition (DAQ) system
        2. These signals are referred to as pulses. An online event builder groups pulses into events using a simplified algorithm to trigger on S1 and S2 candidates and stores a 1 ms window around each trigger. During offline processing by the custom developed data processor PAX, pulses are further segmented into smaller intervals, denoted as hits, by separating individual signals, which may have been grouped into the same pulse waveform.
9.  XMASS
    1. A direct dark matter search in XMASS-I
        1. 
