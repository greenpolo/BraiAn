import braian
import braian.config
from pathlib import Path

# test that read legacy data are the same of data not legacy
root_dir = Path("/home/castoldi/Documents/silvalab_analyses/ethofearless")
config_file = root_dir/"analysis"/"braian_config_4groups.yml"
config = braian.config.BraiAnConfig(config_file)
ss_silvalab = config._ontology.partition("summary structures")
experiment = config.from_csv(legacy=True)
g = experiment[0]
g.to_csv(config.output_dir, overwrite=True, legacy=False)
g_ = braian.AnimalGroup.from_csv(config.output_dir/f"{g.name}_sum.csv", ontology=config._ontology, legacy=False)
assert all(
    (bd.data == bd_.data).all(skipna=True)
    for a,a_ in zip(experiment[0],g_)
    for m,m_ in zip(a.markers, a_.markers)
    for bd,bd_ in zip(a._markers_data[m],a_._markers_data[m_])
)