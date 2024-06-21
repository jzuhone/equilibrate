from yt.fields.field_info_container import FieldInfoContainer

# We need to specify which fields we might have in our dataset.  The field info
# container subclass here will define which fields it knows about.  There are
# optionally methods on it that get called which can be subclassed.

b_units = "code_magnetic"
pres_units = "code_mass/(code_length*code_time**2)"
en_units = "code_mass * (code_length/code_time)**2"
rho_units = "code_mass / code_length**3"


class ClusterGeneratorFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("velocity_x", ("code_length/code_time", ["velx"], None)),
        ("velocity_y", ("code_length/code_time", ["vely"], None)),
        ("velocity_z", ("code_length/code_time", ["velz"], None)),
        ("dens", ("code_mass/code_length**3", ["density"], None)),
        ("temp", ("code_temperature", ["temperature"], None)),
        ("pres", (pres_units, ["pressure"], None)),
        ("gpot", ("code_length**2/code_time**2", ["gravitational_potential"], None)),
        ("magp", (pres_units, ["magnetic_pressure"], None)),
        ("divb", ("code_magnetic/code_length", [], None)),
        ("magx", (b_units, [], "B_x")),
        ("magy", (b_units, [], "B_y")),
        ("magz", (b_units, [], "B_z")),
    )

    known_particle_fields = (
        # Identical form to above
        # ( "name", ("units", ["fields", "to", "alias"], # "display_name")),
    )

    def __init__(self, ds, field_list):
        super().__init__(ds, field_list)
        # If you want, you can check self.field_list

    def setup_fluid_fields(self):
        # Here we do anything that might need info about the dataset.
        # You can use self.alias, self.add_output_field (for on-disk fields)
        # and self.add_field (for derived fields).
        pass

    def setup_particle_fields(self, ptype):
        super().setup_particle_fields(ptype)
        # This will get called for every particle type.
