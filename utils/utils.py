import pandas as pd
import numpy as np
import obonet


class HpoDisease:
    def __init__(self, name=None, name_cn=None, disease_id=None, annotations=None,
                 modes_of_inheritance=None, prevalence=None, onset_age=None, death_age=None):
        self.name = name
        self.name_cn = name_cn
        self.disease_id = disease_id
        self.annotations = annotations
        self.modes_of_inheritance = modes_of_inheritance
        self.prevalence = prevalence
        self.onset_age = onset_age
        self.death_age = death_age


def load_disease_map(phenotype_hpoa_path, hpo_id_to_name):
    phenotype_hpoa = pd.read_csv(phenotype_hpoa_path, sep='\t', header=0, dtype=str, skiprows=4)
    phenotype_hpoa.rename(columns={'#DatabaseID': 'DatabaseID'}, inplace=True)
    orpha_phenotype_hpoa = phenotype_hpoa[phenotype_hpoa['DatabaseID'].str.contains('ORPHA')]           # only ORPHA
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[orpha_phenotype_hpoa['Aspect'].isin(['P'])]         # only PHENOTYPIC_ABNORMALITY
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[~orpha_phenotype_hpoa['Qualifier'].isin(['NOT'])]   # no Excluded
    orpha_phenotype_hpoa = orpha_phenotype_hpoa.loc[:, ['DatabaseID', 'DiseaseName', 'HPO_ID', 'Frequency']]

    all_orpha = list({}.fromkeys(orpha_phenotype_hpoa['DatabaseID'].values).keys())
    orpha_phenotype_hpoa.set_index('DatabaseID', inplace=True)

    disease_map = {}
    for orpha in all_orpha:
        if len(orpha_phenotype_hpoa.loc[orpha].shape) == 1:
            orpha_name = orpha_phenotype_hpoa.loc[orpha, 'DiseaseName']
            phenotypic_abnormalities = [{
                'hpo_id': orpha_phenotype_hpoa.loc[orpha][1],
                'hpo_name': hpo_id_to_name.get(orpha_phenotype_hpoa.loc[orpha][1]),
                'freq_id': orpha_phenotype_hpoa.loc[orpha][2],
                'freq_name': hpo_id_to_name.get(orpha_phenotype_hpoa.loc[orpha][2])
            }]
        else:
            orpha_name = orpha_phenotype_hpoa.loc[orpha, 'DiseaseName'].values[0]
            phenotypic_abnormalities = [{
                'hpo_id': anno[1],
                'hpo_name': hpo_id_to_name.get(anno[1]),
                'freq_id': anno[2],
                'freq_name': hpo_id_to_name.get(anno[2])
            } for anno in orpha_phenotype_hpoa.loc[orpha].values]
        disease_map[orpha] = HpoDisease(
            disease_id=orpha,
            name=orpha_name,
            annotations=phenotypic_abnormalities,
            prevalence=1
        )
    return all_orpha, disease_map


hp_obo_path = '/dataset/hp.obo'
phenotype_hpoa_path = '/dataset/phenotype.hpoa'
hpo_onto_graph = obonet.read_obo(hp_obo_path)
hpo_id_to_name = {id_: data.get('name') for id_, data in hpo_onto_graph.nodes(data=True)}
all_orpha, disease_map = load_disease_map(phenotype_hpoa_path, hpo_id_to_name)