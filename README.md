# pyscf_mbe_pol

A fragment-based approach using many-body expansion (MBE) theory is used for the  QM calculation of liquid water. 

BIM (binary interaction method) had been used for the following paper:
* S. Y. Willow, M. A. Salim, K. S. Kim, and S. Hirata, "Ab Initio Molecular Dynamics of Liquid Water Using Embedded-Fragment Second-Order Many-Body Perturbation Theory Towards its Accurate Property Prediction," Sci. Rep. 5, 14358, (2015).

MBE_POL is a modified MBE method by adding polarization energy into MBE.
The polarization energy using a QM/MM approach is estimated based on
* C. Hensen, J. C. Hermann, K. Nam, S. Ma, J. Gao, and H.-D. Holtje, "A Combined QM/MM Approach to Protein-Ligand Interactions: Polarization Effects of the HIV-1 Protease on Selected High Affinity Inhibitors,"  J. Med. Chem. 47, 6673-6680 (2004).
* S. Y. Willow, B. Xie, J. Lawrence, R. S. Eisenberg, and D. D. L. Minh, "On the polarization of ligands by proteins," Phys. Chem. Chem. Phys. 22, 12044-12057 (2020).

