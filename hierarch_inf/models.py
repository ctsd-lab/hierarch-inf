"""BioCRNpyler construction and SBML export for the mechanistic model."""

import biocrnpyler as bcp
import libsbml


def build_detailed_crn(rates, tx_max_occupancy, tl_max_occupancy, pools,
                       initial_dna_uM):
    """Construct the occupancy-limited gene-expression CRN with GFP maturation."""
    parameter_dictionary = {
        ("multi_tx", None, "max_occ"): tx_max_occupancy,
        ("multi_tl", None, "max_occ"): tl_max_occupancy,
        ("multi_tx", None, "kb"): rates["kb_tx"],
        ("multi_tx", None, "ku"): rates["ku_tx"],
        ("multi_tx", None, "k_iso"): rates["kiso_tx"],
        ("multi_tx", None, "ktx"): rates["ktx"],
        ("multi_tl", None, "kb"): rates["kb_tl"],
        ("multi_tl", None, "ku"): rates["ku_tl"],
        ("multi_tl", None, "k_iso"): rates["kiso_tl"],
        ("multi_tl", None, "ktl"): rates["ktl"],
        ("rna_degradation_mm", None, "kb"): rates["kb_deg"],
        ("rna_degradation_mm", None, "ku"): rates["ku_deg"],
        ("rna_degradation_mm", None, "kdeg"): rates["kdeg"],
        "kdil": rates["kdil"], "kmat": rates["kmat"],
    }
    gene = bcp.DNAassembly(
        name="gfp", promoter="pconst", rbs="rbs_strong",
        protein="GFP_immature",
    )
    fluorescent_gfp = bcp.Protein("GFP")
    gfp_loss = bcp.Dilution(
        mechanism_type="global_mechanism",
        filter_dict={"GFP": True, "GFP_immature": True},
        default_on=False,
    )
    mixture = bcp.TxTlExtract(
        name="exp3", components=[gene, fluorescent_gfp],
        parameter_file=None, parameters=parameter_dictionary,
        global_mechanisms=[gfp_loss],
    )
    mixture.add_mechanisms({
        "transcription": bcp.multi_tx(pol=mixture.rnap.get_species()),
        "translation": bcp.multi_tl(ribosome=mixture.ribosome.get_species()),
    }, overwrite=True)
    crn = mixture.compile_crn()
    maturation = bcp.OneStepPathway()
    crn.add_reactions(maturation.update_reactions(
        precursor=[gene.protein], product=[fluorescent_gfp.get_species()],
        k=mixture.get_parameter(maturation, None, "kmat"),
    ))
    initial_conditions = {
        "dna_gfp": initial_dna_uM,
        **{f"protein_{name}": value for name, value in pools.items()},
    }
    crn.initial_concentration_dict = {
        species: initial_conditions[str(species)]
        for species in crn.species if str(species) in initial_conditions
    }
    return crn


def export_for_bioscrape(crn, raw_path, model_path):
    """Export SBML with the paper parameter names and minute time units."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    crn.write_sbml_file(str(raw_path))
    document = libsbml.readSBMLFromFile(str(raw_path))
    sbml_model = document.getModel()
    # Declare the numerical time unit used by these kinetic rates.
    minute = sbml_model.createUnitDefinition()
    minute.setId("minute")
    unit = minute.createUnit()
    unit.setKind(libsbml.UNIT_KIND_SECOND)
    unit.setExponent(1)
    unit.setScale(0)
    unit.setMultiplier(60.0)
    sbml_model.setTimeUnits("minute")
    species_rename = {
        "dna_gfp": "DNA", "protein_RNAP": "RNAP", "rna_gfp": "mRNA",
        "protein_Ribo": "Ribo", "protein_RNase": "RNase",
        "protein_GFP_immature": "GFP_immature", "protein_GFP": "GFP",
    }
    parameter_rename = {
        "kb__multi_tx": "kb_tx", "ku__multi_tx": "ku_tx",
        "k_iso__multi_tx": "kiso_tx", "ktx__multi_tx": "ktx",
        "kb__multi_tl": "kb_tl", "ku__multi_tl": "ku_tl",
        "k_iso__multi_tl": "kiso_tl", "ktl__multi_tl": "ktl",
        "kb__rna_degradation_mm": "kb_deg",
        "ku__rna_degradation_mm": "ku_deg",
        "kdeg__rna_degradation_mm": "kdeg", "kdil__": "kdil",
        "kmat__": "kmat",
    }
    for old, new in species_rename.items():
        sbml_model.getSpecies(old).setId(new)
        for element in document.getListOfAllElements():
            element.renameSIdRefs(old, new)
    for old, new in parameter_rename.items():
        sbml_model.getParameter(old).setId(new)
        for element in document.getListOfAllElements():
            element.renameSIdRefs(old, new)
    libsbml.writeSBMLToFile(document, str(model_path))
    return species_rename
