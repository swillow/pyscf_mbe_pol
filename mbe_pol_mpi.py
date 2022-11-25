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

AU2KCAL_MOL = HARTREE2J*AVOGADRO/4184.0 # 627.5095


def qm_mol (atm_list, basis):
    return gto.M (atom=atm_list, basis=basis, unit='Bohr', verbose=0)

def get_mbe_pol_energy (atom_types, basis, l_mp2=False):

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
                    pair_list_LR.append ( [im, jm, neigh_list[n]])
                    mol_neigh_EF[im].append ( [jm, neigh_list[n]])
                    mol_neigh_EF[jm].append ( [im, -neigh_list[n]])
                elif rij2[n] < RCUT2_LR:
                    pair_list_LR.append ( [im, jm, neigh_list[n]])
        
        return pair_list_QM, mol_neigh_EF, pair_list_LR



    def get_monomer_mm_list (R, box, atm_chgs, imol_neigh):

        # nsite = 3
        jcel = np.array ( [ [jcel, jcel, jcel] for _, jcel in imol_neigh]).reshape (-1,3)
        jbox = np.einsum('ij,j->ij', jcel, box)
        mm_idx = np.array ( [ [3*jm, 3*jm+1, 3*jm+2] for jm, _ in imol_neigh])
        mm_idx = mm_idx.reshape(-1)
        mm_chgs = atm_chgs[mm_idx]
        mm_crds = R[mm_idx] + jbox 

        return mm_crds, mm_chgs, mm_idx 
    
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
                mm_crds, mm_chgs, mm_idx = get_monomer_mm_list (R, box, chgs_old, mol_neigh[im])

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
        chgs_new = np.zeros (chgs_old.shape,dtype=np.float64)
        jbox = np.zeros ( (nsite, ndim))

        for iter in range(10):
            
            for im in range (nmol):
                qm_idx = np.array([3*im, 3*im+1, 3*im+2])
                mol = get_qm_mol(R, qm_idx, jbox, basis)
                mm_crds, mm_chgs, mm_idx = get_monomer_mm_list (R, box, chgs_old, mol_neigh[im])
                rval = scsmp2_energy_density (mol, l_mp2, mm_crds, mm_chgs)
                
                dm = rval['dm'] # SCS-MP2 Density Matrix
                chgs_new[qm_idx] = esp_atomic_charges (mol, dm, esp_options, verbose=0)

            chgs_diff = chgs_new - chgs_old
            rmsd = np.sqrt ((chgs_diff**2).mean())
            chgs_old = chgs_new.copy()
            
            if rmsd < 0.001:
                break 
        
        return chgs_new 


    def mbe_pol_monomers (R, box, atm_chgs, mol_neigh):
        ndim = 3
        qm_cell = np.zeros ( (nsite,ndim) )

        enr_mon = np.zeros ( (nmol), dtype=np.float64)
        grd_mon = np.zeros ( R.shape, dtype=np.float64)
        enr_pol = 0.0
        grd_pol = np.zeros ( R.shape, dtype=np.float64) 

        for im in range (rank, nmol, nproc):
            qm_idx = np.array([3*im, 3*im+1, 3*im+2])
            mol = get_qm_mol (R, qm_idx, qm_cell, basis)
            mm_crds, mm_chgs, mm_idx = get_monomer_mm_list (
                    R, box, atm_chgs, mol_neigh[im])
            rval = scsmp2_energy_gradient (mol, l_mp2, mm_crds, mm_chgs, l_pol=True)
            
            enr_mon[im] = rval['ener']
            grd_mon[qm_idx] = rval['grad']
            # Because of doubly counting of enr_pol
            enr_pol += 0.5*rval['epol']
            grd_pol[qm_idx] += 0.5*rval['epol_qm_grad']
            grd_pol[mm_idx] += 0.5*rval['epol_mm_grad']

        comm.Barrier()
        rval = {}
        rval['enr_mon'] = comm.allreduce (enr_mon, op=MPI.SUM)
        rval['enr_pol'] = comm.allreduce (enr_pol, op=MPI.SUM)
        rval['grd_mon'] = comm.allreduce (grd_mon, op=MPI.SUM)
        rval['grd_pol'] = comm.allreduce (grd_pol, op=MPI.SUM)

        return rval


    def mbe_dimers (R, box, enr_mon, grd_mon, pair_list):

        if rank == 0:
            print ('num pair list', len(pair_list))

        grd = np.zeros ( R.shape, dtype=np.float64)
        qm_cell = np.zeros ( (2*nsite,3), dtype=np.float64)
        enr = 0.0
        npair = len (pair_list)
        for ip in range (rank, npair, nproc):
            #for im, jm, jcel in pair_list:
            im, jm, jcel = pair_list[ip]
            # E_ij - E_i - E_j
            # ij-mer
            qm_idx = np.array([3*im, 3*im+1, 3*im+2, 3*jm, 3*jm+1, 3*jm+2])
            qm_cell[-nsite:] = np.einsum('j,j->j',jcel, box)

            mol = get_qm_mol (R, qm_idx, qm_cell, basis)
            rval = scsmp2_energy_gradient (mol, l_mp2)
            
            enr += (rval['ener'] - enr_mon[im] - enr_mon[jm])
            grd[qm_idx] += (rval['grad'] - grd_mon[qm_idx])
            
            if rank == 0:
                print ('im jm jcel, enr ', im, jm, jcel, enr)

        comm.Barrier()
        enr = comm.reduce (enr, op=MPI.SUM)
        grd = comm.reduce (grd, op=MPI.SUM)

        return enr, grd 


    def ener_coul_lr (R, box, chgs, pair_list_Coul):

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

        pair_list_qm, mol_neigh_ef, pair_list_lr = get_pair_list (R, box)
        
        if rank == 0:
            print ('start ESP')
        m_atm_chgs = atm_chgs.copy()
        m_atm_chgs = esp_hf_update (R, box, m_atm_chgs, mol_neigh_ef)

        if rank == 0:
            print ('start Ecoul_LR')

        enr_lr, grd_lr = ener_coul_lr (R, box, m_atm_chgs, pair_list_lr)
        
        if rank == 0:
            print ('start MON')
        
        mval = mbe_pol_monomers (R, box, m_atm_chgs, mol_neigh_ef)
        
        if rank == 0:
            print ('start DIM')
        enr_dim, grd_dim = mbe_dimers (R, box, mval['enr_mon'], mval['grd_mon'], pair_list_qm)

        rval = {}
        if rank == 0:
            rval['ener'] = np.sum(mval['enr_mon']) + enr_dim + mval['enr_pol'] + enr_lr
            rval['grad'] = mval['grd_mon'] + grd_dim + mval['grd_pol'] + grd_lr
            rval['chgs'] = m_atm_chgs 

            print ('enr_pol', mval['enr_pol']*AU2KCAL_MOL)
            print ('enr_dim', enr_dim*AU2KCAL_MOL)

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
        
        mbe_pol_fn = get_mbe_pol_energy (atm_types, basis=basis, l_mp2=l_mp2)
        rval = mbe_pol_fn (atm_crds, box, atm_chgs)
    
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
