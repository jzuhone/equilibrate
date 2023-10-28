"""
Catch-all module for various data structures and their construction algorithms.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import Unit, exceptions, unyt_array

from cluster_generator.utils import ensure_list, ensure_ytarray, mylog


def build_yt_dataset_fields(grid, models, domain_dimensions, centers, velocities):
    from cluster_generator.model import ClusterModel

    # -- Segmenting the fields by type -- #
    _added_fields = [
        "density",
        "pressure",
        "dark_matter_density",
        "stellar_density",
        "gravitational_potential",
    ]
    _mass_weighted_fields = ["temperature"]
    _mass_weighted_nonfields = ["velocity_x", "velocity_y", "velocity_z"]
    units = {
        "density": "Msun/kpc**3",
        "pressure": "Msun/kpc/Myr**2",
        "dark_matter_density": "Msun/kpc**3",
        "stellar_density": "Msun/kpc**3",
        "temperature": "K",
        "gravitational_potential": "kpc**2/Myr**2",
        "velocity_x": "kpc/Myr",
        "velocity_y": "kpc/Myr",
        "velocity_z": "kpc/Myr",
        "magnetic_field_strength": "G",
    }

    # -- Sanity Checks -- #
    models = ensure_list(models)

    for mid, model in enumerate(models):
        if isinstance(model, str):
            models[mid] = ClusterModel.from_h5_file(model)

    centers = ensure_ytarray(centers, "kpc")
    velocities = ensure_ytarray(velocities, "kpc/Myr")

    mylog.info("Building yt dataset structure...")

    x, y, z = grid
    fields = _added_fields + _mass_weighted_fields
    data = {}
    for i, p in enumerate(models):
        xx = x - centers.d[i][0]
        yy = y - centers.d[i][1]
        zz = z - centers.d[i][2]
        rr = np.sqrt(xx * xx + yy * yy + zz * zz)
        fd = InterpolatedUnivariateSpline(p["radius"].d, p["density"].d)

        # -- Managing constituent fields -- #
        for field in fields:
            if field not in p:  # We don't have this data, don't do anything.
                continue
            else:
                # Managing units
                try:
                    p[field] = p[field].to(units[field])
                except exceptions.UnitConversionError as error:
                    if field == "temperature":
                        p[field] = p[field].to("K", equivalence="thermal")
                    else:
                        raise error

            if (
                field not in data
            ):  # This data needs to be initialized in the data object.
                data[field] = unyt_array(np.zeros(domain_dimensions), units[field])

            f = InterpolatedUnivariateSpline(p["radius"].d, p[field].d)

            if field in _added_fields:
                data[field] += unyt_array(f(rr), units[field])  # Just add the values
            elif field in _mass_weighted_fields:
                data[field] += unyt_array(
                    f(rr) * fd(rr), Unit(units[field])
                )  # Density weighted values

        # -- Managing Velocities -- #
        for j, field in enumerate(_mass_weighted_nonfields):
            if field not in data:
                data[field] = unyt_array(np.zeros(domain_dimensions), units[field])

            # We do need to add it #
            data[field] += unyt_array(velocities.d[i][j] * fd(rr), Unit(units[field]))

    if "density" in data:
        for field in _mass_weighted_fields + _mass_weighted_nonfields:
            data[field] /= data["density"].d

    return data
