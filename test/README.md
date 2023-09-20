### ```attention!```

The tests developed here are rather sensitive to the initial parameters provided for STORM 3 (by default), that is, files [parameters.py](../parameters.py) and [ProbabilityDensityFunctions_TWO.csv](../model_input/ProbabilityDensityFunctions_TWO.csv) (or [ProbabilityDensityFunctions_ONE-pmf.csv](../model_input/ProbabilityDensityFunctions_ONE-pmf.csv)), from which a copy is provided in this folder (i.e., *copy_parameters.py* and *copy_ProbabilityDensityFunctions_TWO.csv*).\
Therefore, if for some reason tests initially start failing, please replace STORM 3 default files/parameters (having made a security copy first) by the ones provided in this directory.
If the problem persists... then call for help.


### ```requirements```

Tests are passing on both Windows and Linux platforms.
Below there is a list of all the Python libraries necessary (by OS) to run STORM 3 (tests included).\
[dependency-libraries *are not* listed below.]

| Linux (Ubuntu 22.04.3 LTS // 64-bit) | Windows 10 Ed. (v.22H2 // 64-bit) |
| :--- | :--- |
| python_abi=3.11=3_cp311<br>&nbsp;&nbsp;&nbsp;&nbsp;(with standard libraries: ``argparse``, ``datetime``,<br>&nbsp;&nbsp;&nbsp;&nbsp;``functools``, ``operator``, ``os``, ``pathlib``,``sys``,<br>&nbsp;&nbsp;&nbsp;&nbsp;``warnings``, ``zoneinfo``) | python=3.9.16=h4de0772_0_cpython<br>&nbsp;&nbsp;&nbsp;&nbsp;(with standard libraries: ``argparse``, ``datetime``,<br>&nbsp;&nbsp;&nbsp;&nbsp;``functools``, ``operator``, ``os``, ``pathlib``,``sys``,<br>&nbsp;&nbsp;&nbsp;&nbsp;``warnings``, ``zoneinfo``) |
| gdal=3.7.0=py311h281082f_1<br>&nbsp;&nbsp;&nbsp;&nbsp;(called by ``osgeo``) | gdal=3.6.2=py39h3be0312_3<br>&nbsp;&nbsp;&nbsp;&nbsp;(called by ``osgeo``) |
| geopandas=0.13.2=pyhd8ed1ab_1 | geopandas=0.13.2=pyhd8ed1ab_1 |
| geopandas-base=0.13.2=pyha770c72_1 | geopandas-base=0.13.2=pyha770c72_1 |
| netcdf4=1.6.4=mpi_mpich_py311hdfd729c_0 | netcdf4=1.6.0=nompi_py39h34fa13a_103 |
| numpy=1.24.4=py311h64a7726_0 | numpy=1.24.4=py39h816b6a6_0 |
| pandas=2.0.3=py311h320fe9a_1 | pandas=2.0.3=py39h1679cfb_1 |
| pointpats=2.2.0=py_0 | pointpats=2.3.0=pyhd8ed1ab_0 |
| pyarrow=12.0.1=py311h39c9aba_6_cpu | pyarrow=11.0.0=py39hdb2b141_13_cpu |
| pyproj=3.6.0=py311h331fe15_0 | pyproj=3.4.1=py39h9727d73_0 |
| python-dateutil=2.8.2=pyhd8ed1ab_0 | python-dateutil=2.8.2=pyhd8ed1ab_0 |
| rasterio=1.3.7=py311h138ec3c_1 | rasterio=1.3.4=py39hce277b7_0 |
| rioxarray=0.14.1=pyhd8ed1ab_0 | rioxarray=0.14.1=pyhd8ed1ab_0 |
| scikit-image=0.21.0=py311hb755f60_0 | scikit-image=0.21.0=py39h99910a6_0 |
| scikit-learn=1.3.0=py311hc009520_0 | scikit-learn=1.3.0=py39hfa9d973_0 |
| scipy=1.11.1=py311h64a7726_0 | scipy=1.11.1=py39hde5eda1_0 |
| statsmodels=0.14.0=py311h1f0f07a_1 | statsmodels=0.14.0=py39hbaa61f9_1 |
| tqdm=4.64.0=pyhd8ed1ab_0 | tqdm=4.65.0=pyhd8ed1ab_1 |
| vonmisesmixtures==1.0.0<br>&nbsp;&nbsp;&nbsp;&nbsp;(via ``pip`` or manually installed) | vonmisesmixtures==1.0.0<br>&nbsp;&nbsp;&nbsp;&nbsp;(via ``pip`` or manually installed) |
| xarray=2023.7.0=pyhd8ed1ab_0 | xarray=2023.7.0=pyhd8ed1ab_0 |
