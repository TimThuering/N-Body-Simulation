# Preprocessing for the N-Body-Simulation

The `./preprocess` executable can be used to convert asteroid data gather from NASA's JPL [Small-Body Database](https://ssd.jpl.nasa.gov/tools/sbdb_query.html).
This can be done as follows:
1. select the desired orbit class
2. if necessary, add additional object/orbit constraints ("Physical Parameter Fields")
3. select the required output fields (order doesn't matter); these are:
    - IAU name
    - H
    - diameter
    - albedo
    - mass (**if available**)
    - e (eccentricity)
    - a (semi-major axis)
    - i (inclination)
    - peri (argument of perihelion)
    - M (mean anomaly)
    - node (longitude of the ascending node)
    - epoch
    - orbit class
4. download the resulting CSV file

The resulting file contains the asteroids given as Keplerian orbital elements. 
In order to use them in our simulation, they must at first be converted to cartesian state vectors.

## Converting the Keplerian orbital elements to cartesian state vectors

This can be done using our `./preprocess` utility:

```bash
n-body simulation
Usage:
  ./preprocess [OPTION...] positional parameters

  -o, --out arg       the filename to store the processed state vectors (default: state_vectors.csv)
  -d, --diameter arg  the minimal diameter for a main-belt asteroid to be used (in m) (default: 0.0)
  -h, --help          print this helper message
```

An example invocation can be: `./preprocess -o my_simulation.csv data/`.
In this case, the preprocessing utility tries to convert the Keplerian orbital elements in **all** CSV files in the `data/` directory to cartesian state vectors.

The cartesian state vectors CSV file contains ten columns in total: `id, name, class, mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z`.
Since the mass of an object is necessary for the n-body simulation but almost no asteroids have a mass associated with them, 
our preprocessing utility approximates masses according to the following scheme:

1. geometric albedo: to approximate the geometric albedo, we use a random number distribution based on the asteroid's orbit class:
   - AMO (Armoer): [0.450, 0.550]
   - APO (Apollo): [0.450, 0.550]
   - ATE (Aten): [0.450, 0.550]
   - IEO (Interior Earth Objects): [0.450, 0.550]
   - MCA (Mars-crossing): [0.450, 0.550]
   - IMB (Inner Main-belt): [0.030, 0.103]
   - MBA (Main-belt): [0.097, 0.203]
   - OMB (Outer Main-belt): [0.197, 0.5]
   - CEN (Centaur): [0.450, 0.750]
   - TJN (Jupiter Trojans): [0.188, 0.124]
   - TNO (TransNeptunian Objects): [0.022, 0.130]
   - AST (Asteroid): [0.450, 0.550]
   - PAA (Parabolic): [0.450, 0.550]
   - HYA (Hyperbolic): [0.450, 0.550]
2. diameter: if no diameter is given, one is approximated using the asteroids absolute magnitude `H` and geometric `albedo`: d = 1329 * albedo^{-0.5} * 10^{-0.2H}
3. asteroid density: approximate the density of an asteroid given its assumed geometric albedo based on the three main asteroid categories:
   - C-type (chondrtie): most common, probably consist of clay and silicate rocks; assumption: albedo < 0.1 -> p = 1.38 g/cm^3
   - S-type ("stony"): made up of silicate materials and nickel-iron; assumption: 0.1 <= albedo <= 0.2 -> p = 2.71 g/cm^3
   - M-type (nickel-iron): metallic; assumption: albedo > 0.2 -> p = 5.32 g/cm^3
4. mass: finally, approximate the mass assuming a spherical shape: m = 4/3 Pi * (d/2)^3 * p

Two things have to be noted:
1. The already provided `data/planets_and_moons.csv` dataset contains hand-provided data for the planets, dwarf, planets, and named moons in Keplerian orbital elements.
   Note that this file contains an additional column called `central_body` listing the body around which the specific body orbits. This isn't necessary for the JPL data
   since by definition all asteroids orbit the Sun.
2. The Sun isn't present in any Keplerian orbital element data since it is impossible to specify its Keplerian orbital elements with respect to itself. 
   However, our preprocessing utility will always manually add the Sun at the cartesian position (0, 0, 0) with an initial velocity of (0, 0, 0) and a mass of 1.98847 * 10^{30} to the final simulation file.
3. It may happen that Pluto and its moons will not form a stable system, i.e., some moons (mainly Kerberos and Hydra) will be ejected from the system. This is most
   the consequence of insufficient data from JPL’s Horizons website (for more information see https://arxiv.org/pdf/2204.04226.pdf).
