from mpi4py import MPI
from pyscf import gto, scf, qmmm
from pyscf.data.nist import BOHR, HARTREE2J, AVOGADRO
from esp import esp_atomic_charges
import numpy as np
from scsmp2grad import scsmp2_energy_gradient, scsmp2_energy_density


comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()


BOHR2ANG = BOHR
ANG2BOHR = 1.0/BOHR
RCUT_QM = 7.5*ANG2BOHR # Pairwise interaction for Dimers
RCUT_EF = 9.0*ANG2BOHR # Embedding field
RCUT_LR = 12.0*ANG2BOHR # Long-range Coulomb Interaction

RCUT2_QM = RCUT_QM*RCUT_QM
RCUT2_EF = RCUT_EF*RCUT_EF
RCUT2_LR = RCUT_LR*RCUT_LR

#RCUT = 7.5*ANG2BOHR
#RCUT2 = RCUT*RCUT
AU2KCAL_MOL = HARTREE2J*AVOGADRO/4184.0 # 627.5095


def qm_mol (atm_list, basis):
    return gto.M (atom=atm_list, basis=basis, unit='Bohr', verbose=0)

def get_bim_energy (atom_types, basis, l_mp2=False):

    esp_options = {
            "probe"   : 0.7, # A
            "restraint" : True, # False: ESP atomic charges, True: RESP atomic charges
            "resp_hfree" : True, # Exclude hydrogen atoms from the restaining procedure.
            "resp_a"  : 0.001, # au
            "resp_b"  : 0.1,   # au
            "resp_maxiter" : 25, # maximum iteraction
            "resp_tolerance" : 1.0e-4, # e
    }
    atm_types = np.array (atom_types)
    idx1 = np.arange (3) - 1
    neigh_list = np.array([[ic, jc, kc] for ic in idx1
                                    for jc in idx1
                                    for kc in idx1])
    n_neigh = neigh_list.shape[0]

    nsite = 3
    nmol = len(atm_types)//nsite
    mol_pairs = [ [im, jm] for im in range (1, nmol)
                           for jm in range (im-1) ]
    
    
    def get_pair_list (R, box):
        pair_list_QM = []
        pair_list_LR = []
        mol_neigh_EF = {}
        pair_list_CR = []
        
        neigh_cell = np.einsum('ij,j->ij', neigh_list, box)

        for im in range (nmol):
            mol_neigh_EF[im] = []
        
        for im, jm in mol_pairs:
            iO = nsite*im
            jO = nsite*jm 
            pos_ij = R[iO] - (R[jO]+neigh_cell)
            rij2 = np.einsum('ij,ij->i', pos_ij, pos_ij)

            for n in range(n_neigh):
                if rij2[n] < RCUT2_QM:
                    pair_list_QM.append ( [im, jm, neigh_list[n]])
                    mol_neigh_EF[im].append ( [jm, neigh_list[n]])
                    mol_neigh_EF[jm].append ( [im, -neigh_list[n]])
                elif rij2[n] < RCUT2_EF:
                    pair_list_CR.append ( [im, jm, neigh_list[n]])
                    mol_neigh_EF[im].append ( [jm, neigh_list[n]])
                    mol_neigh_EF[jm].append ( [im, -neigh_list[n]])
                elif rij2[n] < RCUT2_LR:
                    pair_list_LR.append ( [im, jm, neigh_list[n]])
        
        return pair_list_QM, mol_neigh_EF, pair_list_LR, pair_list_CR



    def get_monomer_mm_list (R, box, imol_neigh):

        # nsite = 3
        jcel = np.array ( [ [jcel, jcel, jcel] for _, jcel in imol_neigh]).reshape (-1,3)
        jbox = np.einsum('ij,j->ij', jcel, box)
        mm_idx = np.array ( [ [3*jm, 3*jm+1, 3*jm+2] for jm, _ in imol_neigh])
        mm_idx = mm_idx.reshape(-1)
        mm_crds = R[mm_idx] + jbox 

        return mm_crds, mm_idx 


    def get_dimer_mm_list (R, box, im, jm, jcel, mol_neigh):

        mm_idx = []
        mm_cel = []

        for km, kcel in mol_neigh[im]:
            if km == jm and (kcel == jcel).all():
                continue 
            mm_idx.append ([3*km, 3*km+1, 3*km+2])
            mm_cel.append ([kcel, kcel, kcel])

        icel0 = np.array([0, 0, 0])
        for km, kcel in mol_neigh[jm]:
            kcel0 = kcel + jcel #[kcel[i]+jcel[i] for i in range(3)]
            
            if (km == im) and (kcel0 == icel0).all():
                continue 

            l_add = True
            for lm, lcel in mol_neigh[im]:
                if km == lm and (kcel0 == lcel).all():
                    l_add = False
                    break
                
            if l_add:
                mm_idx.append ([3*km, 3*km+1, 3*km+2])
                mm_cel.append ([kcel0, kcel0, kcel0])
            
        mm_idx = np.array(mm_idx).reshape(-1)
        mm_cel = np.array(mm_cel).reshape(-1,3)
        mm_box = np.einsum('ij,j->ij', mm_cel, box)

        mm_crds = R[mm_idx] + mm_box

        return mm_crds, mm_idx
        


    def get_qm_mol (atm_crds, qm_idx, qm_cell, basis):
        
        qm_natom = qm_idx.shape[0]
        qm_atnm = atm_types[qm_idx]
        qm_xyz = atm_crds[qm_idx] + qm_cell

        atm_list = [ [qm_atnm[i], qm_xyz[i]] for i in range(qm_natom) ]

        return qm_mol (atm_list, basis)



    def esp_hf_update (R, box, atm_chgs, mol_neigh):
        # atomic point charges are estated at HF
        

        ndim = 3
        chgs_old = atm_chgs.copy()
        jbox = np.zeros ( (nsite, ndim))

        for _ in range(10):
            chgs_new = np.zeros (chgs_old.shape,dtype=np.float64)
            for im in range (rank, nmol, nproc):
                qm_idx = np.array([3*im, 3*im+1, 3*im+2])
                mol = get_qm_mol(R, qm_idx, jbox, basis)
                mm_crds, mm_idx = get_monomer_mm_list (R, box, mol_neigh[im])
                mm_chgs = chgs_old[mm_idx]
                mf = scf.RHF(mol)
                mf.chkfile = None 
                mf = mf.run(verbose=0)
                mf_qmmm = qmmm.mm_charge (mf, mm_crds, mm_chgs, unit='Bohr').run()
                dm = mf_qmmm.make_rdm1() # RHF Density Matrix
                
                chgs_new[qm_idx] = esp_atomic_charges (mol, dm, esp_options, verbose=0)

            comm.Barrier()
            chgs_new = comm.allreduce (chgs_new, op=MPI.SUM)
            chgs_diff = chgs_new - chgs_old
            rmsd = np.sqrt ((chgs_diff**2).mean())
            chgs_old = chgs_new.copy()

            if rmsd < 0.001:
                break 
        
        return chgs_new 

    def esp_update (R, box, atm_chgs, mol_neigh):
        # atomic point charges are estated at SCS-MP2
        
        ndim = 3
        chgs_old = atm_chgs.copy()
        jbox = np.zeros ( (nsite, ndim))

        for iter in range(10):
            chgs_new = np.zeros (chgs_old.shape,dtype=np.float64)
            for im in range (rank, nmol, nproc):
                qm_idx = np.array([3*im, 3*im+1, 3*im+2])
                mol = get_qm_mol(R, qm_idx, jbox, basis)
                mm_crds, mm_idx = get_monomer_mm_list (R, box, mol_neigh[im])
                mm_chgs = chgs_old[mm_idx]
                rval = scsmp2_energy_density (mol, l_mp2, mm_crds, mm_chgs)
                
                dm = rval['dm'] # SCS-MP2 Density Matrix
                chgs_new[qm_idx] = esp_atomic_charges (mol, dm, esp_options, verbose=0)

            comm.Barrier()
            chgs_new = comm.allreduce (chgs_new, op=MPI.SUM)
            chgs_diff = chgs_new - chgs_old
            rmsd = np.sqrt ((chgs_diff**2).mean())
            chgs_old = chgs_new.copy()
            print ('iter', iter, rmsd)
            if rmsd < 0.001:
                break 
        
        return chgs_new 




    def bim_monomers (R, box, atm_chgs, mol_neigh):
        """
        monomer : QM
        embedding field : MM
        """

        ndim = 3
        grd = np.zeros ( R.shape, dtype=np.float64)
        qm_cell = np.zeros ( (nsite,ndim) )

        enr = 0.0
        for im in range (rank, nmol, nproc):
            qm_idx = np.array([3*im, 3*im+1, 3*im+2])
            mol = get_qm_mol (R, qm_idx, qm_cell, basis)
            mm_crds, mm_idx = get_monomer_mm_list (
                R, box, mol_neigh[im])
            mm_chgs = atm_chgs[mm_idx]
            rval = scsmp2_energy_gradient (mol, l_mp2, mm_crds, mm_chgs)
            
            enr += rval['ener']
            grd[qm_idx] += rval['grad']
            grd[mm_idx] += rval['mm_grad']

        comm.Barrier()
        enr = comm.allreduce (enr, op=MPI.SUM)
        grd = comm.allreduce (grd, op=MPI.SUM)

        return enr, grd 


    def bim_dimers (R, box, atm_chgs, mol_neigh, pair_list):

        if rank == 0:
            print ('num pair list', len(pair_list))

        grd = np.zeros ( R.shape, dtype=np.float64)
        qm_cell = np.zeros ( (2*nsite,3), dtype=np.float64)
        enr = 0.0
        npair = len(pair_list)
        for ip in range (rank, npair, nproc):
            
            im, jm, jcel = pair_list[ip]
            # E_ij - E_i - E_j
            # ij-mer
            qm_idx = np.array([3*im, 3*im+1, 3*im+2, 3*jm, 3*jm+1, 3*jm+2])
            qm_cell[-nsite:] = np.einsum('j,j->j',jcel, box)

            mol = get_qm_mol (R, qm_idx, qm_cell, basis)
            mm_crds, mm_idx = \
                get_dimer_mm_list (R, box, im, jm, jcel, mol_neigh)
            mm_chgs = atm_chgs[mm_idx]
            rval = scsmp2_energy_gradient (mol, l_mp2, mm_crds, mm_chgs)
            
            enr += rval['ener']
            grd[qm_idx] += rval['grad']
            grd[mm_idx] += rval['mm_grad']

            # i-mer
            imol = get_qm_mol (R, qm_idx[:nsite], qm_cell[:nsite], basis)
            # Add Jmol positions into BQ
            jm_xyz = R[qm_idx[-nsite:]] + qm_cell[-nsite:]
            ibq_crds = np.array(list(mm_crds) + list(jm_xyz))
            ibq_idx = np.array(list(mm_idx)+list(qm_idx[-nsite:]))
            ibq_chgs = atm_chgs[ibq_idx]
            rval = scsmp2_energy_gradient (imol, l_mp2, ibq_crds, ibq_chgs)
            
            enr -= rval['ener']
            grd[qm_idx[:nsite]] -= rval['grad']
            grd[ibq_idx] -= rval['mm_grad']

            # j-mer
            jmol = get_qm_mol (R, qm_idx[-nsite:], qm_cell[-nsite:], basis)
            
            im_xyz = R[qm_idx[:nsite]]
            # Add Imol positions into BQ
            jbq_crds = np.array(list(mm_crds) + list(im_xyz))
            # Add Jmol charges into BQ
            jbq_idx = np.array(list(mm_idx)+list(qm_idx[:nsite]))
            jbq_chgs = atm_chgs[jbq_idx]
            rval = scsmp2_energy_gradient (jmol, l_mp2, jbq_crds, jbq_chgs)
            
            enr -= rval['ener']
            grd[qm_idx[-nsite:]] -= rval['grad']
            grd[jbq_idx] -= rval['mm_grad']
            
            if rank == 0:
                print ('im jm jcel, enr ', im, jm, jcel, enr)

        comm.Barrier()
        enr = comm.allreduce (enr, op=MPI.SUM)
        grd = comm.allreduce (grd, op=MPI.SUM)

        return enr, grd 


    def ener_coul_mm (R, box, chgs, pair_list_Coul):

        enr = 0.0
        grd = np.zeros ( R.shape, dtype=np.float64 )
        npair = len (pair_list_Coul)
        for ip in range (rank, npair, nproc):
            #for im, jm, jcel in pair_list_Coul:
            im, jm, jcel = pair_list_Coul[ip]
            jbox = np.einsum('j,j->j', jcel, box)
            for ia in [3*im, 3*im+1, 3*im+2]:
                for ja in [3*jm, 3*jm+1, 3*jm+2]:
                    dRij = R[ia] - (R[ja]+jbox)
                    rij2 = np.einsum('j,j->', dRij, dRij)
                    rij  = np.sqrt (rij2)
                    enr_coul = chgs[ia]*chgs[ja]/rij
                    grd_coul = -enr_coul*dRij/rij2
                    enr += enr_coul
                    grd[ia] += grd_coul 
                    grd[ja] -= grd_coul 
        
        comm.Barrier()
        enr = comm.reduce (enr, op=MPI.SUM)
        grd = comm.reduce (grd, op=MPI.SUM)
        
        return enr, grd 


    def compute (R, box, atm_chgs):
        pair_list_qm, mol_neigh_ef, pair_list_lr, pair_list_cr = \
            get_pair_list (R, box)
        
        if rank == 0:
            print ('start ESP')
        
        m_atm_chgs = atm_chgs.copy()
        m_atm_chgs = esp_hf_update (R, box, m_atm_chgs, mol_neigh_ef)
        
        if rank == 0:
            print ('start Ecoul_LR')

        enr_lr, grd_lr = ener_coul_mm (R, box, m_atm_chgs, pair_list_lr)
        
        if rank == 0:
            print ('start Ener Correction')
        enr_cr, grd_cr = ener_coul_mm (R, box, m_atm_chgs, pair_list_cr)

        if rank == 0:
            print ('start MON')
        enr_mon, grd_mon = bim_monomers (R, box, m_atm_chgs, mol_neigh_ef)
        
        if rank == 0:
            print ('start DIM')
        enr_dim, grd_dim = bim_dimers (R, box, m_atm_chgs, mol_neigh_ef, pair_list_qm)

        rval = {}
        if rank == 0:
            rval['ener'] = enr_mon + enr_dim + enr_lr - enr_cr
            rval['grad'] = grd_mon + grd_dim + grd_lr - grd_cr
            rval['chgs'] = m_atm_chgs 

        return rval 

    return compute 


if __name__ == '__main__':
    import xyz
    import json
    import sys 

    fname_json = 'input.json'
    if len(sys.argv) > 1:
        fname_json = sys.argv[1]
    
    with open (fname_json) as f:
        data = json.load(f)

        fname_xyz = data['fname_xyz'] 
        atm_types, atm_crds = xyz.read_xyz (fname_xyz)
        atm_crds = atm_crds * ANG2BOHR

        box = np.array (data['box'])*ANG2BOHR
        atm_chgs = np.zeros (atm_crds.shape[0])
        basis = data['basis']
        l_mp2 = data['l_mp2']
    
        bim_fn = get_bim_energy (atm_types, basis=basis, l_mp2=l_mp2)
        rval = bim_fn (atm_crds, box, atm_chgs)
    
        if rank == 0:
            enr  = rval['ener']
            grds = rval['grad']
            atm_chgs = rval['chgs']

            fname_grd = data['fname_grd']
            fout = open (fname_grd, 'w', 1)
            for grd in grds:
                val = grd*AU2KCAL_MOL/BOHR2ANG
                print (val[0], val[1], val[2], file=fout)


    MPI.Finalize()