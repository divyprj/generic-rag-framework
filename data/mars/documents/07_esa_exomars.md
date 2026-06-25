# ESA ExoMars Program

The ExoMars (Exobiology on Mars) program is the European Space Agency's flagship Mars exploration initiative, designed to address one of the most fundamental questions in planetary science: has life ever existed on Mars? Originally conceived as a joint ESA-NASA endeavor, the program's history has been marked by ambitious science goals, international partnership shifts, significant technical challenges, and hard-won lessons that continue to shape European planetary exploration.

## Program History and Structure

The ExoMars program was initiated by ESA's Ministerial Council in 2005 as a cornerstone mission for the Aurora exploration program. Initially planned as a single mission, it evolved into a two-phase program:

- **Phase 1 (2016):** Trace Gas Orbiter (TGO) and Schiaparelli Entry, Descent and Landing Demonstrator Module (EDM).
- **Phase 2 (originally 2018, repeatedly delayed):** Rosalind Franklin rover with a landing platform.

NASA was the original partner for Phase 2, but budget constraints led NASA to withdraw in 2012. ESA subsequently partnered with Roscosmos (Russia), which agreed to provide Proton launch vehicles and the landing platform. This partnership was dissolved in March 2022 following Russia's invasion of Ukraine, forcing ESA to restructure the mission with European solutions. Thales Alenia Space, based in Turin, Italy, serves as the prime industrial contractor for both mission phases.

## Trace Gas Orbiter (TGO, 2016)

### Launch and Orbital Insertion

TGO launched on March 14, 2016, aboard a Proton-M/Breeze-M rocket from Baikonur Cosmodrome, Kazakhstan. It entered Mars orbit on October 19, 2016, and subsequently performed a year-long aerobraking campaign to lower its orbit from a highly elliptical capture orbit to its science orbit at approximately 400 kilometers altitude, achieved in April 2018.

### Scientific Instruments

TGO carries four main instruments:

- **NOMAD (Nadir and Occultation for Mars Discovery):** A suite of three spectrometers (solar occultation, limb/nadir, and ultraviolet-visible) that maps the atmospheric composition, searching for trace gases including methane, water vapor, and other hydrocarbons. NOMAD can detect methane at concentrations as low as 50 parts per trillion by volume.
- **ACS (Atmospheric Chemistry Suite):** Provided by the Russian Space Research Institute (IKI), ACS complements NOMAD with three infrared spectrometers for detailed atmospheric profiling, measuring temperature, aerosol properties, and trace gas abundances.
- **CaSSIS (Colour and Stereo Surface Imaging System):** A high-resolution colour stereo camera producing images at 4.5 meters per pixel, designed to characterize surface features related to trace gas sources and sinks.
- **FREND (Fine Resolution Epithermal Neutron Detector):** A neutron detector that maps subsurface hydrogen (a proxy for water ice) to a depth of approximately 1 meter with spatial resolution better than previously achieved from orbit.

### Key Findings

TGO's most significant and surprising result has been the non-detection of methane. Despite earlier reports of methane by Mars Express's PFS instrument and Curiosity's SAM instrument (which measured background levels of ~0.41 ppbv), NOMAD and ACS have placed an upper limit on global methane abundance of approximately 0.05 parts per billion by volume—far below previous detections. This discrepancy remains one of the most debated puzzles in Mars science, with possible explanations including rapid methane destruction at the surface, highly localized and transient emissions, or systematic measurement differences between instruments.

TGO also serves as a vital communications relay, providing data relay services for Curiosity, InSight (during its operational lifetime), and Perseverance. It is expected to serve as the primary relay for the Rosalind Franklin rover.

## Schiaparelli EDM Lander (2016)

### Mission and Design

Schiaparelli (named after the 19th-century Italian astronomer Giovanni Schiaparelli) was a 577-kilogram technology demonstrator designed to prove ESA's capability to land on Mars. It separated from TGO on October 16, 2016, and began its descent to Meridiani Planum on October 19, 2016.

### Entry, Descent, and Landing Failure

Schiaparelli's descent began nominally. The heat shield withstood entry temperatures exceeding 1,500 °C, and the parachute deployed at an altitude of approximately 11 kilometers. However, during the parachute phase, the lander's Inertial Measurement Unit (IMU) experienced saturation—the IMU data exceeded its maximum measurable range during an unexpected oscillation period. This caused the onboard computer to calculate a negative altitude (i.e., it believed it was below the surface). As a result, the parachute and back shell were jettisoned prematurely at an altitude of approximately 3.7 kilometers, and the braking thrusters fired for only 3 seconds instead of the planned 30 seconds. Schiaparelli struck the surface at an estimated 540 kilometers per hour, creating a dark scar visible in MRO HiRISE images.

### Lessons Learned

The independent inquiry board identified several contributing factors: insufficient testing of the IMU under the specific oscillation conditions encountered during parachute descent, incomplete modeling of parachute dynamics, and inadequate software safeguards against physically impossible altitude calculations. ESA incorporated these lessons into the design of the Rosalind Franklin rover's landing system, including enhanced IMU dynamic range, improved parachute deployment sequencing, additional sensor cross-checking, and software constraints preventing negative altitude calculations.

## Rosalind Franklin Rover

### Design and Specifications

The Rosalind Franklin rover—named after the British chemist and X-ray crystallographer whose work was pivotal to understanding DNA structure—is designed to search for biosignatures in the Martian subsurface. The rover has a mass of approximately 310 kilograms, measures about 2.0 meters long, 1.2 meters wide, and 2.0 meters tall (with mast deployed), and is powered by solar panels.

### Unique Drilling Capability

Rosalind Franklin's most distinctive feature is its drill, capable of extracting samples from up to 2 meters below the Martian surface—far deeper than any other Mars mission. Subsurface samples are scientifically critical because the Martian surface is bombarded by ultraviolet radiation and permeated by oxidizing compounds (such as the perchlorates discovered by Phoenix) that would destroy organic molecules and potential biosignatures. At depths of 2 meters, samples are shielded from these destructive processes, potentially preserving billions-of-years-old organic chemistry.

### Pasteur Instrument Suite

The rover carries the Pasteur payload, named after the pioneering microbiologist Louis Pasteur, comprising nine instruments:

- **PanCam:** A panoramic camera system for geological context and navigation.
- **ISEM (Infrared Spectrometer for ExoMars):** Identifies mineral targets for close-up investigation.
- **CLUPI (Close-Up Imager):** A high-resolution camera for textural analysis of rocks and drill cores.
- **WISDOM (Water Ice Subsurface Deposit Observation on Mars):** A ground-penetrating radar to guide drill placement.
- **Adron:** A neutron spectrometer for detecting subsurface hydrogen.
- **Ma_MISS (Mars Multispectral Imager for Subsurface Studies):** A spectrometer integrated into the drill to analyze borehole walls.
- **MicrOmega:** A visible-infrared hyperspectral microscope for analyzing drill samples.
- **RLS (Raman Laser Spectrometer):** Identifies minerals and detects organic compounds.
- **MOMA (Mars Organic Molecule Analyser):** The primary astrobiology instrument, combining a gas chromatograph, mass spectrometer, and laser desorption system capable of detecting a wide range of organic molecules.

### Autonomy System

Rosalind Franklin incorporates advanced autonomous navigation capabilities, allowing it to traverse up to 70 meters per sol without ground-in-the-loop commands. The rover's navigation system uses stereo cameras to build 3D terrain maps and autonomously plan safe paths around obstacles—essential given the 4–24 minute one-way communication delay between Earth and Mars.

### Mission Timeline and Challenges

The Rosalind Franklin rover was originally scheduled for launch in 2018, then delayed to 2020 due to parachute qualification issues and landing platform readiness. A further delay to 2022 was caused by additional parachute testing failures and the COVID-19 pandemic. The dissolution of the ESA-Roscosmos partnership in 2022 forced ESA to find alternative solutions for the landing platform (originally Russia's Kazachok platform) and the launch vehicle (originally a Proton rocket). ESA's Ministerial Council in November 2022 approved continued development with European and international partners, including NASA, which agreed to provide critical subsystems. The current launch target is no earlier than 2028, with landing planned on Oxia Planum—a region rich in phyllosilicates (clay minerals) and interpreted as an ancient river delta environment.
