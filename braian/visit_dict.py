# for breadth-first search
def pop_queue(q):
    x = q[0]
    return q[1:], x

# for depth-first search
def pop_stack(s):
    x = s[-1]
    return s[:-1], x

# Breadth-first search
def find_subtree(node, key, value, children_key):
    if key in node and node[key] == value:
        return node

    to_visit = node[children_key]
    while len(to_visit) != 0:
        to_visit, child = pop_queue(to_visit)
        if child[key] == value:
            return child
        to_visit.extend(child[children_key])
    return None

def add_child(node, child, children_key):
    node[children_key].append(child)
    return None

def remove_child(node, child, children_key):
    node[children_key].remove(child)
    return None


# Breadth-first visit
def visit_bfs(node, children_key, fun):
    to_visit = [node]
    depth = 0
    while len(to_visit) != 0:
        level_size = len(to_visit)
        while level_size != 0:
            level_size -= 1
            to_visit, child = pop_queue(to_visit)
            fun(child, depth)
            to_visit.extend(child[children_key])
        depth += 1

# Depth-first visit
def visit_dfs(node, children_key, fun):
    to_visit = [node]
    depth = 0
    while len(to_visit) != 0:
        level_size = len(to_visit)
        while level_size != 0:
            level_size -= 1
            to_visit, child = pop_stack(to_visit)
            fun(child, depth)
            to_visit.extend(reversed(child[children_key]))
        depth += 1

# a modified DFS
def non_overlapping_where(node, children_key, filter_fun, mode="bfs") -> list[dict]:
    assert mode in ("bfs", "dfs"), "Invalid mode!"
    cut = []
    to_visit = [node]
    depth = 0
    while len(to_visit) != 0:
        level_size = len(to_visit)
        while level_size != 0:
            level_size -= 1
            to_visit, child = pop_stack(to_visit) if mode == "dfs" else pop_queue(to_visit)
            if filter_fun(child, depth):
                cut.append(child)
            else:
                to_visit.extend(reversed(child[children_key])) if mode == "dfs" else to_visit.extend(child[children_key])
        depth += 1
    return cut

def add_boolean_attribute(tree, children_key, attr_name, is_true, visit=visit_bfs):
    def select(node, depth):
        node[attr_name] = is_true(node, depth)
    visit(tree, children_key, select)

def del_attribute(tree, children_key, attr_name, visit=visit_bfs):
    def del_attr(node, depth):
        del node[attr_name]
    visit(tree, children_key, del_attr)

def get_where(tree, children_key, filter_fun, visit=visit_bfs) -> list[dict]:
    nodes = []
    add_if_true = lambda node,depth: nodes.append(node) if filter_fun(node,depth) else None
    visit(tree, children_key, add_if_true)
    return nodes

def visit_parents(tree, children_key, fun, visit=visit_bfs):
    global parents
    parents = [None]
    def record_parent(node, depth):
        global parents
        parents, parent = pop_queue(parents)
        fun(parent, node)
        n_children = len(node[children_key])
        parents.extend([node]*n_children)
    visit(tree, children_key, record_parent)

def get_parents_where(tree, children_key, filter_fun, nodes_key):
    parents = dict()
    add_if_true = lambda parent,node: parents.__setitem__(node[nodes_key], parent) if filter_fun(parent,node) else None
    visit_parents(tree, children_key, add_if_true)
    return parents

def is_leaf(node, children_key):
    return len(node[children_key]) == 0

def get_leaves(node, children_key):
    return get_where(node, children_key, lambda x,d: is_leaf(x, children_key))

def get_all_nodes(tree, children_key, visit=visit_bfs):
    def remove_children(node):
        c = node.copy()
        c.pop(children_key)
        return c
    nodes = []
    visit(tree, children_key, lambda n,d: nodes.append(remove_children(n)))
    return nodes

def prune_where(tree, children_key, fun):
    to_visit = [tree]
    while len(to_visit) != 0:
        to_visit, node = pop_queue(to_visit)
        node[children_key] = [child for child in node[children_key] if not fun(child)]
        to_visit.extend(node[children_key])

if __name__ == "__main__":
    import json
    PATH = "./data/AllenMouseBrainOntology.json"
    with open(PATH, "r") as file:
        data = json.load(file)
    tree = data["msg"][0]
    #key = "acronym"
    #value = "FRP"
    #subtree = find_subtree(tree, key, value, "children")
    #print("SUBTREE:", subtree, "\n")
    #leaves = get_leaves(subtree, "children")
    #print("LEAVES:", leaves, "\n")
    #nodes = get_all_nodes(subtree, "children")
    #print("NODES:", nodes)
    level = 3
    level_regions = get_where(tree, "children", lambda x,d: x["st_level"] == level)
    asd = [region["acronym"] for region in level_regions]
    print(f"NODES OF LEVEL {level}: {asd}")

    subtree = find_subtree(tree, "acronym", "Isocortex", "children")
    all_discrepancies_parents = get_where(subtree, "children", lambda x,d: any([child["st_level"] != x["st_level"]+1 for child in x["children"]]))
    all_discrepancies = [(child["acronym"], parent["st_level"], child["st_level"]) for parent in all_discrepancies_parents for child in parent["children"] if child["st_level"] != parent["st_level"]+1]
    summary_sts = ["FRP", "MOp", "MOs", "SSp-n", "SSp-bfd", "SSp-ll", "SSp-m", "SSp-ul", "SSp-tr", "SSp-un", "SSs", "GU", "Prosv-sr", "AUDd", "AUDp", "AUDpo", "AUDv", "VISal", "VISam", "VISl", "VISp", "VISpl", "VISpm", "VISli", "VISpor", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl", "AId", "AIp", "AIv", "RSPagl", "RSPd", "RSPv", "VISa", "VISrl", "TEa", "PERI", "ECT", "MOB", "AOB", "AON", "TT", "DP", "PIR", "NLOT", "COAa", "COAp", "PAA", "TR", "CA1", "CA2", "CA3", "DG", "FC", "IG", "ENTl", "ENTm", "PAR", "POST", "PRE", "SUB", "ProS", "HATA", "APr", "CLA", "EPd", "EPv", "LA", "BLA", "BMA", "PA", "CP", "ACB", "FS", "OT", "LSc", "LSr", "LSv", "SF", "SH", "AAA", "BA", "CEA", "IA", "MEA", "GPe", "GPi", "SI", "MA", "MS", "NDB", "TRS", "BST", "BAC", "VAL", "VM", "VPL", "VPLpc", "VPM", "VPMpc", "PoT", "SPFm", "SPFp", "SPA", "PP", "MG", "LGd", "LP", "PO", "POL", "SGN", "AV", "AM", "AD", "IAM", "IAD", "LD", "IMD", "MD", "SMT", "PR", "PVT", "PT", "RE", "Xi", "RH", "CM", "PCN", "CL", "PF", "PIL", "RT", "IGL", "IntG", "LGv", "SubG", "MH", "LH", "SO", "ASO", "PVH", "PVa", "PVi", "ARH", "ADP", "AVP", "AVPV", "DMH", "MEPO", "MPO", "OV", "PD", "PS", "PVp", "PVpo", "SBPV", "SCH", "SFO", "VMPO", "VLPO", "AHN", "LM", "MM", "SUM", "TMd", "TMv", "MPN", "PMd", "PMv", "PVHd", "VMH", "PH", "LHA", "LPO", "PST", "PSTN", "PeF", "RCH", "STN", "TU", "ZI", "ME", "SCs", "IC", "NB", "SAG", "PBG", "MEV", "SCO", "SNr", "VTA", "PN", "RR", "MRN", "SCm", "PAG", "APN", "MPT", "NOT", "NPC", "OP", "PPT", "CUN", "RN", "III", "MA3", "EW", "IV", "Pa4", "VTN", "AT", "LT", "DT", "MT", "SNc", "PPN", "IF", "IPN", "RL", "CLI", "DR", "NLL", "PSV", "PB", "SOC", "B", "DTN", "PDTg", "PCG", "PG", "PRNc", "SG", "SUT", "TRN", "V", "P5", "Acs5", "PC5", "I5", "CS", "LC", "LDT", "NI", "PRNr", "RPO", "SLC", "SLD", "AP", "DCO", "VCO", "CU", "GR", "ECU", "NTB", "NTS", "SPVC", "SPVI", "SPVO", "Pa5", "VI", "VII", "ACVII", "AMB", "DMX", "GRN", "ICB", "IO", "IRN", "ISN", "LIN", "LRN", "MARN", "MDRN", "MDRNd", "MDRNv", "PARN", "PAS", "PGRNd", "PGRNl", "NR", "PRP", "PPY", "LAV", "MV", "SPIV", "SUV", "x", "XII", "y", "RM", "RPA", "RO", "LING", "CENT", "CUL", "DEC", "FOTU", "PYR", "UVU", "NOD", "SIM", "AN", "PRM", "COPY", "PFL", "FL", "FN", "IP", "DN", "VeCB", "fiber tracts"]
    # summary_sts = ["FRP1", "FRP2/3", "FRP5", "FRP6a", "FRP6b", "MOp1", "MOp2/3", "MOp5", "MOp6a", "MOp6b", "MOs1", "MOs2/3", "MOs5", "MOs6a", "MOs6b", "SSp-n1", "SSp-n2/3", "SSp-n4", "SSp-n5", "SSp-n6a", "SSp-n6b", "SSp-bfd1", "SSp-bfd2/3", "SSp-bfd4", "SSp-bfd5", "SSp-bfd6a", "SSp-bfd6b", "SSp-ll1", "SSp-ll2/3", "SSp-ll4", "SSp-ll5", "SSp-ll6a", "SSp-ll6b", "SSp-m1", "SSp-m2/3", "SSp-m4", "SSp-m5", "SSp-m6a", "SSp-m6b", "SSp-ul1", "SSp-ul2/3", "SSp-ul4", "SSp-ul5", "SSp-ul6a", "SSp-ul6b", "SSp-tr1", "SSp-tr2/3", "SSp-tr4", "SSp-tr5", "SSp-tr6a", "SSp-tr6b", "SSp-un1", "SSp-un2/3", "SSp-un4", "SSp-un5", "SSp-un6a", "SSp-un6b", "SSs1", "SSs2/3", "SSs4", "SSs5", "SSs6a", "SSs6b", "GU1", "GU2/3", "GU4", "GU5", "GU6a", "GU6b", "VISC1", "VISC2/3", "VISC4", "VISC5", "VISC6a", "VISC6b", "AUDd1", "AUDd2/3", "AUDd4", "AUDd5", "AUDd6a", "AUDd6b", "AUDp1", "AUDp2/3", "AUDp4", "AUDp5", "AUDp6a", "AUDp6b", "AUDpo1", "AUDpo2/3", "AUDpo4", "AUDpo5", "AUDpo6a", "AUDpo6b", "AUDv1", "AUDv2/3", "AUDv4", "AUDv5", "AUDv6a", "AUDv6b", "VISal1", "VISal2/3", "VISal4", "VISal5", "VISal6a", "VISal6b", "VISam1", "VISam2/3", "VISam4", "VISam5", "VISam6a", "VISam6b", "VISl1", "VISl2/3", "VISl4", "VISl5", "VISl6a", "VISl6b", "VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b", "VISpl1", "VISpl2/3", "VISpl4", "VISpl5", "VISpl6a", "VISpl6b", "VISpm1", "VISpm2/3", "VISpm4", "VISpm5", "VISpm6a", "VISpm6b", "VISli1", "VISli2/3", "VISli4", "VISli5", "VISli6a", "VISli6b", "VISpor1", "VISpor2/3", "VISpor4", "VISpor5", "VISpor6a", "VISpor6b", "ACAd1", "ACAd2/3", "ACAd5", "ACAd6a", "ACAd6b", "ACAv1", "ACAv2/3", "ACAv5", "ACAv6a", "ACAv6b", "PL1", "PL2/3", "PL5", "PL6a", "PL6b", "ILA1", "ILA2/3", "ILA5", "ILA6a", "ILA6b", "ORBl1", "ORBl2/3", "ORBl5", "ORBl6a", "ORBl6b", "ORBm1", "ORBm2/3", "ORBm5", "ORBm6a", "ORBm6b", "ORBvl1", "ORBvl2/3", "ORBvl5", "ORBvl6a", "ORBvl6b", "AId1", "AId2/3", "AId5", "AId6a", "AId6b", "AIp1", "AIp2/3", "AIp5", "AIp6a", "AIp6b", "AIv1", "AIv2/3", "AIv5", "AIv6a", "AIv6b", "RSPagl1", "RSPagl2/3", "RSPagl5", "RSPagl6a", "RSPagl6b", "RSPd1", "RSPd2/3", "RSPd5", "RSPd6a", "RSPd6b", "RSPv1", "RSPv2/3", "RSPv5", "RSPv6a", "RSPv6b", "VISa1", "VISa2/3", "VISa4", "VISa5", "VISa6a", "VISa6b", "VISrl1", "VISrl2/3", "VISrl4", "VISrl5", "VISrl6a", "VISrl6b", "TEa1", "TEa2/3", "TEa4", "TEa5", "TEa6a", "TEa6b", "PERI6a", "PERI6b", "PERI1", "PERI5", "PERI2/3", "ECT1", "ECT2/3", "ECT5", "ECT6a", "ECT6b", "MOB", "AOBgl", "AOBgr", "AOBmi", "AON", "TTd", "TTv", "DP", "PIR", "NLOT1", "NLOT2", "NLOT3", "COAa", "COApl", "COApm", "PAA", "TR", "CA1", "CA2", "CA3", "DG-mo", "DG-po", "DG-sg", "FC", "IG", "ENTl1", "ENTl2", "ENTl3", "ENTl5", "ENTl6a", "ENTm1", "ENTm2", "ENTm3", "ENTm5", "ENTm6", "PAR", "POST", "PRE", "SUB", "ProS", "HATA", "APr", "CLA", "EPd", "EPv", "LA", "BLAa", "BLAp", "BLAv", "BMAa", "BMAp", "PA", "CP", "ACB", "FS", "OT", "LSc", "LSr", "LSv", "SF", "SH", "AAA", "BA", "CEAc", "CEAl", "CEAm", "IA", "MEA", "GPe", "GPi", "SI", "MA", "MS", "NDB", "TRS", "BST", "BAC", "VAL", "VM", "VPL", "VPLpc", "VPM", "VPMpc", "PoT", "SPFm", "SPFp", "SPA", "PP", "MGd", "MGv", "MGm", "LGd-sh", "LGd-co", "LGd-ip", "LP", "PO", "POL", "SGN", "Eth", "AV", "AMd", "AMv", "AD", "IAM", "IAD", "LD", "IMD", "MD", "SMT", "PR", "PVT", "PT", "RE", "Xi", "RH", "CM", "PCN", "CL", "PF", "PIL", "RT", "IGL", "IntG", "LGv", "SubG", "MH", "LH", "SO", "ASO", "PVH", "PVa", "PVi", "ARH", "ADP", "AVP", "AVPV", "DMH", "MEPO", "MPO", "OV", "PD", "PS", "PVp", "PVpo", "SBPV", "SCH", "SFO", "VMPO", "VLPO", "AHN", "LM", "Mmme", "Mml", "Mmm", "Mmp", "Mmd", "SUM", "TMd", "TMv", "MPN", "PMd", "PMv", "PVHd", "VMH", "PH", "LHA", "LPO", "PST", "PSTN", "PeF", "RCH", "STN", "TU", "ZI", "FF", "ME", "SCop", "SCsg", "SCzo", "ICc", "ICd", "ICe", "NB", "SAG", "PBG", "MEV", "SCO", "SNr", "VTA", "PN", "RR", "MRN", "SCdg", "SCdw", "SCiw", "SCig", "PAG", "PRC", "INC", "ND", "Su3", "APN", "MPT", "NOT", "NPC", "OP", "PPT", "RPF", "CUN", "RN", "III", "MA3", "EW", "IV", "Pa4", "VTN", "AT", "LT", "DT", "MT", "SNc", "PPN", "IF", "IPR", "IPC", "IPA", "IPL", "IPI", "IPDM", "IPDL", "IPRL", "RL", "CLI", "DR", "NLL", "PSV", "PB", "KF", "POR", "SOCm", "SOCl", "B", "DTN", "PDTg", "PCG", "PG", "PRNc", "SG", "SUT", "TRN", "V", "P5", "Acs5", "PC5", "I5", "CS", "LC", "LDT", "NI", "PRNr", "RPO", "SLC", "SLD", "AP", "DCO", "VCO", "CU", "GR", "ECU", "NTB", "NTS", "SPVC", "SPVI", "SPVO", "Pa5", "VI", "VII", "ACVII", "AMBd", "AMBv", "DMX", "GRN", "ICB", "IO", "IRN", "ISN", "LIN", "LRNm", "LRNp", "MARN", "MDRNd", "MDRNv", "PARN", "PAS", "PGRNd", "PGRNl", "NR", "PRP", "PPY", "LAV", "MV", "SPIV", "SUV", "x", "XII", "y", "RM", "RPA", "RO", "LING", "CENT2", "CENT3", "CUL4, 5", "DEC", "FOTU", "PYR", "UVU", "NOD", "SIM", "ANcr1", "ANcr2", "PRM", "COPY", "PFL", "FL", "FN", "IP", "DN", "VeCB", "von", "onl", "lot", "lotd", "aco", "IIn", "bsc", "csc", "och", "opt", "IIIn", "mlf", "pc", "IVn", "moV", "sV", "sptV", "VIIn", "gVIIn", "vVIIIn", "tb", "das", "ll", "cic", "bic", "ts", "cuf", "ml", "cbc", "scp", "dscp", "uf", "sctv", "mcp", "icp", "sctd", "arb", "scwm", "cc", "fa", "ec", "ee", "ccg", "fp", "ccb", "ccs", "cst", "int", "cpd", "py", "pyd", "em", "or", "ar", "nst", "tspd", "dtd", "tspc", "rust", "vtd", "amc", "act", "cing", "alv", "df", "fi", "mct", "fx", "dhc", "vhc", "st", "stc", "mfb", "sup", "pm", "mtt", "mtg", "mp", "sm", "frroot", "grey", "CH", "CTX", "CTXpl", "Isocortex", "FRP", "MO", "MOp", "MOs", "SS", "SSp", "SSp-n", "SSp-bfd", "SSp-ll", "SSp-m", "SSp-ul", "SSp-tr", "SSp-un", "SSs", "GU", "Prosv-sr", "AUD", "AUDd", "AUDp", "AUDpo", "AUDv", "VIS", "VISal", "VISam", "VISl", "VISp", "VISpl", "VISpm", "VISli", "VISpor", "ACA", "ACAd", "ACAv", "PL", "ILA", "ORB", "ORBl", "ORBm", "ORBvl", "AI", "AId", "AIp", "AIv", "RSP", "RSPagl", "RSPd", "RSPv", "PTLp", "VISa", "VISrl", "TEa", "PERI", "ECT", "OLF", "AOB", "TT", "NLOT", "COA", "COAp", "HPF", "HIP", "CA", "DG", "RHP", "ENT", "ENTl", "ENTm", "CTXsp", "EP", "BLA", "BMA", "CNU", "STR", "STRd", "STRv", "LSX", "LS", "sAMY", "CEA", "PAL", "PALd", "PALv", "PALm", "MSC", "PALc", "BS", "IB", "TH", "DORsm", "VENT", "VP", "SPF", "GENd", "MG", "LGd", "DORpm", "LAT", "ATN", "AM", "MED", "MTN", "ILM", "GENv", "EPI", "HY", "PVZ", "PVR", "MEZ", "MBO", "MM", "TM", "LZ", "MB", "MBsen", "SCs", "IC", "MBmot", "SCm", "PRT", "MBsta", "RAmb", "IPN", "HB", "P", "P-sen", "SOC", "P-mot", "P-sat", "MY", "MY-sen", "CN", "DCN", "MY-mot", "AMB", "LRN", "MDRN", "PGRN", "PHY", "VNC", "MY-sat", "CB", "CBX", "VERM", "CENT", "CUL", "HEM", "AN", "CBN", "fiber tracts", "cm", "In", "lotg", "Vn", "VIIIn", "cVIIIn", "Xn", "drt", "cett", "dc", "cbf", "cbp", "lfbs", "lfbst", "eps", "epsc", "tsp", "mfbs", "mfbc", "fxs", "fxpo", "hc", "mfsbshy", "mfbsma", "mfbse", "VS", "hbc", "VL", "SEZ", "chpl", "V3", "AQ", "V4", "V4r", "c"]
    discrepancies = [(discrepancy,nl,pl) for (discrepancy,nl,pl) in all_discrepancies if discrepancy not in summary_sts]
    print(discrepancies)
    #print([summary_st for summary_st in summary_sts if summary_st not in [r[0] for r in all_discrepancies]])
    print(f"N TOTAL DISCREPANCIES: {len(all_discrepancies)}")
    print(f"N SUMMARY STRUCTURES: {len(summary_sts)}")
    #print(f"N DISCREPANCIES - SUMMARY STRUCTURES: {len(discrepancies) - len(summary_sts)}")
    print(f"N DISCREPANCIES: {len(discrepancies)}")